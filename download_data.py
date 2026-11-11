#!/usr/bin/env python3
"""
download_data.py  --  Dataset stage for:
  "Intelligent Industrial Fan Fault Detection using Acoustic Spectrogram Analysis
   with Lightweight MobileNetV2 and Real-Time Edge Deployment on Raspberry Pi 4"

REAL DATA ONLY.  The project never generates or substitutes artificial sound: if the dataset cannot be
obtained, this script stops with exit code 2 and says how to fix it.

Where the data comes from (first match wins)
  1. dataset/<machine>/ already populated                    -> reused
  2. --data-dir  folder you already have (extracted DCASE 2020 / MIMII), copied, never moved
  3. --archive   dev_data_<machine>.zip you downloaded yourself
  4. dataset/_downloads/dev_data_<machine>.zip, if present
  5. Zenodo (DCASE 2020 Task 2 development set, record 3678171)  -> resumable download
  6. Kaggle (kagglehub, --kaggle-slug owner/dataset)           -> optional

Final layout
  dataset/<machine>/train/normal/      healthy clips for training (DCASE releases no faulty clip for training)
  dataset/<machine>/test/normal/       healthy clips held out for testing
  dataset/<machine>/test/abnormal/     faulty clips (test only)
  dataset/<machine>/manifest.csv       path,label,split,machine_id   (label 0=healthy, 1=faulty)
  dataset/<machine>/DATASET_SOURCE.json

Usage:  python download_data.py                              # download from Zenodo
        python download_data.py --archive dev_data_fan.zip   # zip you downloaded yourself
        python download_data.py --data-dir D:/MIMII/fan      # folder you already have
        python download_data.py --stats-only                 # statistics + figure for the existing data
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Global audio / feature configuration (reused by the later training scripts)
# --------------------------------------------------------------------------- #
SR = 16000                 # DCASE 2020 / MIMII sampling rate
DURATION = 10.0            # seconds per clip
N_SAMPLES = int(SR * DURATION)
N_FFT = 1024
HOP = 512
N_MELS = 128
# With center=False: 1 + (160000 - 1024) // 512 = 311 frames -> (128, 311)
EXPECTED_MEL_SHAPE = (N_MELS, 313)

DEFAULT_ROOT = Path("dataset/fan")
FIG_DIR = Path("outputs/figures")

MACHINES = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]   # DCASE 2020 Task 2 dev set


# md5 checksums published on https://zenodo.org/records/3678171 (DCASE 2020 Task 2 development dataset)
ZENODO_MD5 = {"fan": "649bdfc06263ae7a838963f43b6641e6", "pump": "90e7091ef722b7238a7f1009365779cd",
              "slider": "da24a757719f0d94d5aa2d646bbfdc86", "ToyCar": "4dec75ca8d9f666aa9e4c1894a740501",
              "ToyConveyor": "03b6aa1bfd09a39d53af0ba39f71fa91", "valve": "34d2672f55bb041589ef79c10dc89934"}


def check_md5(path: Path, machine: str):
    """True / False if the file matches the official checksum, None if unknown."""
    import hashlib
    want = ZENODO_MD5.get(machine)
    if not want:
        return None
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest() == want


def zenodo_urls(machine: str = "fan"):
    return [f"https://zenodo.org/records/3678171/files/dev_data_{machine}.zip?download=1",
            f"https://zenodo.org/record/3678171/files/dev_data_{machine}.zip?download=1"]


ZENODO_URLS = [
    # Official DCASE 2020 Task 2 development data, fan machine type.
    "https://zenodo.org/records/3678171/files/dev_data_fan.zip?download=1",
    "https://zenodo.org/record/3678171/files/dev_data_fan.zip?download=1",
    # Kept because the project brief names .tar.gz; Zenodo normally ships .zip.
    "https://zenodo.org/record/3678171/files/dev_data_fan.tar.gz?download=1",
]


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# 1. Audio loading
# --------------------------------------------------------------------------- #
def load_audio(path, sr: int = SR, mono: bool = True):
    """Load a wav file -> (waveform float32 in [-1,1], sampling_rate).

    * multi-channel files (raw MIMII is 8-ch) are averaged to mono
    * resampled to `sr` if the file rate differs
    """
    import soundfile as sf

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        wav, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:  # unusual codec -> let librosa/audioread try
        import librosa
        wav, file_sr = librosa.load(str(path), sr=None, mono=False)
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.T
    if wav.ndim > 1 and mono:
        # microphone arrays (original MIMII: 8 channels): first channel only, exactly as the DCASE 2020 data were made;
        # stereo files are averaged
        wav = wav[:, 0] if wav.shape[1] > 2 else wav.mean(axis=1)
    if file_sr != sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
        file_sr = sr
    return wav.astype(np.float32), file_sr


def fix_length(wav: np.ndarray, n: int = N_SAMPLES) -> np.ndarray:
    """Pad with zeros / crop so every clip is exactly 10 s."""
    if len(wav) >= n:
        return wav[:n]
    return np.pad(wav, (0, n - len(wav)))


def wav_to_logmel(wav: np.ndarray, sr: int = SR) -> np.ndarray:
    """10 s waveform -> log-Mel spectrogram of shape (128, 313), dB scale (shared NumPy implementation)."""
    from features_np import logmel
    return logmel(wav)


# --------------------------------------------------------------------------- #
# 2. Dataset presence / manifest
# --------------------------------------------------------------------------- #
def _wavs(folder: Path):
    return sorted(folder.glob("*.wav")) if folder.is_dir() else []


def dataset_ready(root: Path, min_files: int = 20) -> bool:
    return (len(_wavs(root / "train" / "normal")) >= min_files
            and len(_wavs(root / "test" / "normal")) + len(_wavs(root / "test" / "abnormal")) >= min_files)


def read_source(root: Path) -> str:
    f = root / "DATASET_SOURCE.json"
    if f.is_file():
        try:
            return json.loads(f.read_text()).get("source", "unknown")
        except Exception:
            pass
    return "unknown"


def write_source(root: Path, source: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "DATASET_SOURCE.json").write_text(json.dumps({"source": source}, indent=2))


def _machine_id(text: str) -> str:
    m = re.search(r"id[_-]?(\d+)", text.lower())
    return m.group(1) if m else "00"


def write_manifest(root: Path) -> Path:
    rows = []
    for split in ("train", "test"):
        for cls, label in (("normal", 0), ("abnormal", 1)):
            for f in _wavs(root / split / cls):
                rows.append((f.relative_to(root).as_posix(), label, split, _machine_id(f.name)))
    out = root / "manifest.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "label", "split", "machine_id"])
        w.writerows(rows)
    return out


# --------------------------------------------------------------------------- #
# 3. Real data: download (resumable) + extract + re-organise
# --------------------------------------------------------------------------- #
def _download(url: str, dest: Path, retries: int = 3, timeout: int = 30) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            done = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            if done:
                req.add_header("Range", f"bytes={done}-")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resumed = done > 0 and r.status == 206
                if not resumed:
                    done = 0
                total = int(r.headers.get("Content-Length", 0)) + done
                if total:
                    free = shutil.disk_usage(dest.parent).free
                    if free < 2.2 * total:
                        log(f"  [warn] only {free/1e9:.1f} GB free; need ~{2.2*total/1e9:.1f} GB")
                with open(part, "ab" if resumed else "wb") as fh:
                    last = time.time()
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 2:
                            pct = f"{100*done/total:5.1f}%" if total else ""
                            log(f"  downloading {done/1e6:9.1f} MB {pct}")
                            last = time.time()
            part.replace(dest)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 416 and part.exists():       # already fully downloaded
                part.replace(dest)
                return True
            log(f"  [attempt {attempt}/{retries}] HTTP {e.code} for {url}")
            if e.code in (403, 404):
                return False                           # retrying will not help
        except Exception as e:                         # DNS, timeout, reset...
            log(f"  [attempt {attempt}/{retries}] {type(e).__name__}: {e}")
        time.sleep(2 * attempt)
    return False


def _safe_extract(archive: Path, out: Path) -> bool:
    out.mkdir(parents=True, exist_ok=True)
    out_r = out.resolve()
    try:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as z:
                for n in z.namelist():                 # block path traversal
                    if not (out_r / n).resolve().is_relative_to(out_r):
                        raise ValueError(f"unsafe path in archive: {n}")
                z.extractall(out)
            return True
        if tarfile.is_tarfile(archive):
            with tarfile.open(archive) as t:
                for m in t.getmembers():
                    if not (out_r / m.name).resolve().is_relative_to(out_r):
                        raise ValueError(f"unsafe path in archive: {m.name}")
                t.extractall(out)
            return True
        log("  archive is neither zip nor tar (corrupt download?)")
    except Exception as e:
        log(f"  extraction failed: {e}")
    return False


def _classify(p: Path):
    """-> (label 0/1/None, split 'train'/'test'/None, machine_id, dcase_style)"""
    name = p.name.lower()
    parts = [x.lower() for x in p.parts]
    mid = _machine_id("/".join(parts[-3:]))
    if name.startswith("normal_"):
        return 0, ("train" if "train" in parts else "test"), mid, True
    if name.startswith("anomaly_") or name.startswith("abnormal_"):
        return 1, "test", mid, True
    if "abnormal" in parts:                            # raw MIMII layout id_xx/abnormal/
        return 1, None, mid, False
    if "normal" in parts:
        return 0, None, mid, False
    return None, None, mid, False


def pick_snr(files, snr=None):
    """Original MIMII ships -6 / 0 / 6 dB versions with identical file names. Use exactly one of them."""
    tags = {}
    for p in files:
        for part in p.parts:
            m = re.match(r"^(-?\d+)_?dB", part)
            if m:
                tags.setdefault(m.group(1), []).append(p)
    if not tags:
        return files, None
    if len(tags) == 1 and snr is None:
        t = next(iter(tags))
        log(f"  noise level (SNR) of this data: {t} dB")
        return tags[t], t
    want = str(snr) if snr is not None else ("6" if "6" in tags else sorted(tags)[0])
    if want not in tags:
        log(f"  requested SNR {want} dB not found; available: {sorted(tags)} dB")
        return [], None
    log(f"  several noise versions found ({', '.join(sorted(tags))} dB): using {want} dB only "
        f"(6 dB is the least noisy version; change with --snr)")
    return tags[want], want


def organise(src: Path, root: Path, copy: bool = False, snr=None) -> int:
    n = 0
    files, _ = pick_snr(sorted(src.rglob("*.wav")), snr)
    for p in files:
        label, split, mid, dcase = _classify(p)
        if label is None:
            continue
        if split is None:   # raw MIMII: deterministic 80/20 split of normals, abnormals -> test
            h = zlib.crc32(f"{mid}{p.stem}".encode()) % 5
            split = "train" if (label == 0 and h != 0) else "test"
        if label == 1:
            split = "test"
        cls = "normal" if label == 0 else "abnormal"
        new_name = p.name if dcase else f"{cls}_id_{mid}_{p.stem}.wav"
        dest = root / split / cls / new_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            (shutil.copy2 if copy else shutil.move)(str(p), str(dest))
        n += 1
    return n


def _online(host: str = "https://zenodo.org", timeout: int = 6) -> bool:
    try:
        urllib.request.urlopen(urllib.request.Request(host, headers={"User-Agent": "Mozilla/5.0"}),
                               timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True                                    # server answered -> we are online
    except Exception:
        return False


def try_zenodo(root: Path, timeout: int, machine: str = "fan", archive_path: str | None = None, snr=None) -> bool:
    log(f"\n[1/3] Trying Zenodo (DCASE 2020 Task 2, {machine})...")
    dl_dir = root.parent / "_downloads"
    raw_dir = root.parent / "_raw"
    jobs = []
    if archive_path:                                     # user supplied the archive manually
        jobs.append((None, Path(archive_path)))
    else:
        local = dl_dir / f"dev_data_{machine}.zip"
        if local.exists():                               # previously downloaded / manually placed
            jobs.append((None, local))
        elif not _online():
            log("  Zenodo not reachable (no internet / blocked).")
            return False
        else:
            jobs += [(u, dl_dir / u.split("/")[-1].split("?")[0]) for u in zenodo_urls(machine)]
    for url, archive in jobs:
        if url:
            log(f"  URL: {url}")
        else:
            log(f"  using local archive: {archive}")
        if not archive.exists() and not _download(url, archive, timeout=timeout):
            continue
        good = check_md5(archive, machine)
        if good is True:
            log("  checksum (md5) matches the official Zenodo file.")
        elif good is False:
            if url:                                         # our own download is damaged -> fetch again
                log("  checksum does NOT match the official file: the download is damaged; deleting it.")
                archive.unlink(missing_ok=True)
                continue
            log("  note: the md5 of your zip differs from the official Zenodo file (a re-zipped or partial copy?). "
                "Continuing, but if you downloaded it, download it again.")
        if not _safe_extract(archive, raw_dir):
            if url:
                archive.unlink(missing_ok=True)        # corrupt download -> fetch again next time
            continue
        n = organise(raw_dir, root, snr=snr)
        shutil.rmtree(raw_dir, ignore_errors=True)
        if n and dataset_ready(root):
            write_source(root, "dcase2020")
            log(f"  Zenodo OK: {n} wav files organised.")
            return True
    log("  Zenodo unavailable.")
    return False


def try_kaggle(root: Path, slug: str | None) -> bool:
    log("\n[2/3] Trying Kaggle (kagglehub)...")
    if not _online("https://www.kaggle.com"):
        log("  Kaggle not reachable. Skipping.")
        return False
    slug = slug or os.environ.get("KAGGLE_MIMII_SLUG")
    if not slug:
        log("  No Kaggle slug given (use --kaggle-slug owner/dataset or env KAGGLE_MIMII_SLUG). Skipping.")
        return False
    try:
        import kagglehub
        path = Path(kagglehub.dataset_download(slug))
        n = organise(path, root, copy=True)
        if n and dataset_ready(root):
            write_source(root, "kaggle")
            log(f"  Kaggle OK: {n} wav files organised.")
            return True
        log("  Kaggle dataset downloaded but no usable fan wav files found.")
    except Exception as e:
        log(f"  Kaggle failed: {type(e).__name__}: {e}")
    return False


def use_local_dir(root: Path, folder: str, snr=None) -> bool:
    log(f"\n[local] using your dataset folder: {folder}")
    p = Path(folder)
    if not p.is_dir():
        log("  not a folder.")
        return False
    n = organise(p, root, copy=True, snr=snr)        # copy: your original files stay where they are
    if n and dataset_ready(root):
        write_source(root, "user_dataset")
        log(f"  {n} wav files organised.")
        return True
    log("  no usable fan recordings found (expected names like normal_id_00_*.wav / anomaly_id_00_*.wav, or "
        "id_00/normal/*.wav and id_00/abnormal/*.wav).")
    return False


DECK_COUNTS = dict(train_normal=1003, test_normal=100, test_abnormal=341)     # id_00, as stated in the project deck


def deck_check(root: Path, machine: str) -> None:
    if machine != "fan":
        return
    got = dict(train_normal=sum(1 for f in _wavs(root / "train" / "normal") if _machine_id(f.name) == "00"),
               test_normal=sum(1 for f in _wavs(root / "test" / "normal") if _machine_id(f.name) == "00"),
               test_abnormal=sum(1 for f in _wavs(root / "test" / "abnormal") if _machine_id(f.name) == "00"))
    log("\n id_00 counts   : " + ", ".join(f"{k} {got[k]} (deck {DECK_COUNTS[k]})" for k in got))
    if got != DECK_COUNTS:
        log(" note: these differ from the numbers on the deck's slides; put the REAL counts above on the slides.")


# --------------------------------------------------------------------------- #
# 5. Statistics
# --------------------------------------------------------------------------- #
def print_stats(root: Path) -> None:
    import soundfile as sf

    source = read_source(root)
    log("\n" + "=" * 62)
    log(f" DATASET REPORT   (source: {source.upper()})")
    log("=" * 62)
    srs, chans = Counter(), Counter()
    counts, dur = {}, 0.0
    for split in ("train", "test"):
        for cls in ("normal", "abnormal"):
            files = _wavs(root / split / cls)
            counts[(split, cls)] = len(files)
            for f in files:
                try:
                    info = sf.info(str(f))
                    srs[info.samplerate] += 1
                    chans[info.channels] += 1
                    dur += info.frames / info.samplerate
                except Exception as e:
                    log(f"  [warn] unreadable file {f.name}: {e}")
    total = sum(counts.values())
    n_norm = counts[("train", "normal")] + counts[("test", "normal")]
    n_abn = counts[("train", "abnormal")] + counts[("test", "abnormal")]
    log(f" Total files      : {total}")
    for (split, cls), c in counts.items():
        log(f"   {split:5s}/{cls:8s}: {c}")
    log(f" Total duration   : {dur:.0f} s = {dur/3600:.2f} h   (avg {dur/max(total,1):.2f} s/clip)")
    by_id = Counter()
    for split in ("train", "test"):
        for cls in ("normal", "abnormal"):
            for f_ in _wavs(root / split / cls):
                by_id[(_machine_id(f_.name), split, cls)] += 1
    for mid_ in sorted({k[0] for k in by_id}):
        log(f"   id_{mid_}: " + ", ".join(f"{sp}/{c} {by_id[(mid_, sp, c)]}" for sp in ("train", "test")
                                          for c in ("normal", "abnormal") if by_id[(mid_, sp, c)]))
    log(f" Sample rate(s)   : {dict(srs)}")
    log(f" Channels         : {dict(chans)}")
    if n_abn:
        log(f" Class imbalance  : normal:abnormal = {n_norm}:{n_abn} = {n_norm/n_abn:.2f} : 1 "
            f"(abnormal = {100*n_abn/total:.1f}% of all clips)")
        te_n, te_a = counts[("test", "normal")], counts[("test", "abnormal")]
        if te_a:
            log(f" Test-set ratio   : {te_n}:{te_a} = {te_n/te_a:.2f} : 1")
    else:
        log(" Class imbalance  : no abnormal files found!")
    log("=" * 62)


# --------------------------------------------------------------------------- #
# 6. Visualisation
# --------------------------------------------------------------------------- #
def visualize(root: Path, seed: int = 0) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa.display

    normals = _wavs(root / "test" / "normal") or _wavs(root / "train" / "normal")
    abnormals = _wavs(root / "test" / "abnormal")
    if not normals or not abnormals:
        log("[viz] need at least one normal and one abnormal file; skipped.")
        return None
    rng = np.random.default_rng(seed)
    picks = {"Normal": normals[rng.integers(len(normals))],
             "Abnormal": abnormals[rng.integers(len(abnormals))]}

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(14, 7))
    for col, (title, path) in enumerate(picks.items()):
        wav, sr = load_audio(path)
        mel = wav_to_logmel(wav, sr)
        t = np.arange(len(wav)) / sr
        ax[0, col].plot(t, wav, lw=0.4)
        ax[0, col].set(title=f"{title} - waveform ({path.name})", xlabel="Time (s)", ylabel="Amplitude")
        ax[0, col].set_xlim(0, t[-1])
        img = librosa.display.specshow(mel, sr=sr, hop_length=HOP, x_axis="time", y_axis="mel",
                                       fmax=sr / 2, ax=ax[1, col], vmin=-80, vmax=0)
        ax[1, col].set_title(f"{title} - log-Mel spectrogram {mel.shape[0]}x{mel.shape[1]}")
        fig.colorbar(img, ax=ax[1, col], format="%+2.0f dB")
    src = read_source(root).upper()
    fig.suptitle(f"Fan sound: normal vs abnormal  [dataset source: {src}]", fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "normal_vs_abnormal.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    log(f"[viz] saved {out}")
    return out


# --------------------------------------------------------------------------- #
# 7. Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Fan dataset: download / organise / inspect (real data only)")
    ap.add_argument("--machine", default="fan", choices=MACHINES, help="DCASE 2020 Task 2 machine type")
    ap.add_argument("--root", default=None, help="default: dataset/<machine>")
    ap.add_argument("--data-dir", default=None, help="folder you already have (extracted DCASE 2020 / MIMII)")
    ap.add_argument("--archive", default=None, help="path to a manually downloaded dev_data_<machine>.zip")
    ap.add_argument("--snr", default=None, help="for original MIMII with several noise versions: 6, 0 or -6 (default 6)")
    ap.add_argument("--kaggle-slug", default=None, help="owner/dataset for kagglehub (optional)")
    ap.add_argument("--force", action="store_true", help="re-create even if the dataset exists")
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--real-only", action="store_true", help="accepted for compatibility: this is always the case")
    args = ap.parse_args()

    root = Path(args.root or f"dataset/{args.machine}")
    log(f"Dataset root: {root.resolve()}")

    if root.exists() and read_source(root) == "synthetic":      # left over from an older version of the project
        log("Found a leftover ARTIFICIAL dataset from an older version: removing it (this project uses real data only).")
        shutil.rmtree(root)
        shutil.rmtree(root.parent / "cache", ignore_errors=True)
    if args.force and root.exists() and not args.stats_only:
        shutil.rmtree(root)

    if dataset_ready(root):
        log(f"Dataset already present (source: {read_source(root)}). Skipping download.")
    elif args.stats_only:
        log("No dataset found and --stats-only given.")
        return 1
    else:
        if args.data_dir:
            ok = use_local_dir(root, args.data_dir, args.snr)
        else:
            ok = try_zenodo(root, args.timeout, args.machine, args.archive, args.snr) or try_kaggle(root, args.kaggle_slug)
        if not ok:
            log("\nERROR: the real dataset could not be obtained.\n"
                "  * Check your internet connection and run again, or\n"
                f"  * download dev_data_{args.machine}.zip from https://zenodo.org/records/3678171 and run:\n"
                f"      python download_data.py --machine {args.machine} --archive path/to/dev_data_{args.machine}.zip\n"
                "  * or, if you already have the extracted folder:\n"
                f"      python download_data.py --machine {args.machine} --data-dir path/to/folder")
            return 2

    man = write_manifest(root)
    log(f"Manifest written: {man}")
    print_stats(root)
    deck_check(root, args.machine)
    visualize(root)
    log("\nDone. Next step: python echolab_train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
