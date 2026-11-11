# Hardware build & Raspberry Pi setup

## 1. Bill of materials

| Item | Qty | Notes |
|---|---|---|
| Raspberry Pi 4 Model B (2 GB or 4 GB) | 1 | 64-bit Raspberry Pi OS (Bookworm) |
| Official 5.1 V / 3 A USB-C power supply | 1 | Under-voltage causes throttling and inflates latency |
| microSD card 16-32 GB, A1/A2 | 1 | |
| USB microphone (USB Audio Class, plug-and-play) | 1 | Omnidirectional; add a foam wind-shield |
| Red LED + green LED (5 mm) | 1 + 1 | |
| 330 Ω resistor | 2 | LED current limiting |
| Active buzzer (5 V) | 1 | Driven through a transistor, see below |
| NPN transistor 2N2222 / BC547 | 1 | Buzzer switch |
| 1 kΩ resistor | 1 | Transistor base |
| Momentary push-button | 1 | "Acknowledge / mute" |
| Breadboard + jumper wires, heat-sink case | 1 | A heat-sink or fan case avoids thermal throttling |

**Cost.** Fill in the Rs. column from real quotes before submission. A Raspberry Pi 4 on its own is
usually priced well above a "Pi + mic = Rs. 4,500" budget once the power supply, SD card and case
are included, so recompute the total bill of materials honestly and compare *that* to the
vibration-sensor cost. A rigorous cost comparison is stronger than a rounded one.

## 2. Wiring (BCM numbering)

| Function | Pi GPIO | Physical pin | Connection |
|---|---|---|---|
| Green "OK" LED | GPIO 22 | 15 | pin 15 → 330 Ω → LED anode (+); LED cathode (−) → GND pin 14 |
| Red "ALARM" LED | GPIO 17 | 11 | pin 11 → 330 Ω → LED anode (+); LED cathode (−) → GND pin 9 |
| Buzzer driver | GPIO 27 | 13 | pin 13 → 1 kΩ → transistor **base** |
| Ack button | GPIO 23 | 16 | pin 16 → button → GND pin 20 (internal pull-up enabled in software) |
| Buzzer supply | 5 V | 2 | pin 2 → buzzer (+) |
| Microphone | — | USB 2.0 port | any USB port |

```
 GPIO22 (pin15) ──[330Ω]──►|── GND (pin14)          green LED
 GPIO17 (pin11) ──[330Ω]──►|── GND (pin9)           red LED

 5V (pin2) ───────────── Buzzer (+)
                         Buzzer (−) ──── Collector ┐
 GPIO27 (pin13) ──[1kΩ]────────────── Base        │ 2N2222 / BC547
                         GND (pin25) ─ Emitter ────┘

 GPIO23 (pin16) ──── push-button ──── GND (pin20)
```

Electrical limits: GPIO is **3.3 V logic**, max ≈ 16 mA per pin. Never connect the 5 V buzzer
directly to a GPIO pin. LEDs at 330 Ω draw ≈ 4 mA: safe.

## 3. Microphone placement (this affects accuracy more than the model does)

* 0.5-1 m from the fan housing, aimed at the bearing side; **same position every time** — the
  model learns the sound *as recorded from that place*.
* Not in the airflow (wind noise); use a foam wind-shield.
* Do not fix the mic rigidly to a vibrating frame (structure-borne noise); use a rubber mount.
* Record ≥ 1-2 weeks of *normal* operation on site and re-run threshold tuning (Section 6).

## 4. Raspberry Pi OS setup

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y python3-venv python3-pip libportaudio2 libsndfile1 rsync
python3 -m venv --system-site-packages ~/fanenv      # --system-site-packages: uses apt's gpiozero/lgpio
source ~/fanenv/bin/activate
pip install -r requirements_pi.txt
```

If `pip install tflite-runtime` has no wheel for your Python version, use Google's successor package
`pip install ai-edge-litert` — `tflite_runner.py` detects either automatically.

### Copy the project from your PC

```bash
# on the PC, after `python echolab_train.py`
rsync -av models/echolab features_np.py mel_basis.npy tflite_runner.py analyze.py echolab_core.py \
      pi_inference.py benchmark_latency.py requirements_pi.txt pi@<pi-ip>:~/fan/
```
The Pi needs only NumPy + TFLite (no TensorFlow): it runs the same quantised model, PCA files and fused score as the website.

## 5. Check the microphone

```bash
arecord -l                                   # note the card number of the USB mic
arecord -D plughw:1,0 -f S16_LE -r 16000 -c 1 -d 5 test.wav && aplay test.wav
alsamixer                                    # F6 → select the USB card, set capture gain ≈ 60-80 %
python pi_inference.py --list-devices        # index to pass as --device
```

Clipping check: `python -c "import soundfile as sf,numpy as np;x,_=sf.read('test.wav');print(abs(x).max())"`
should print well below 1.0.

## 6. Run

```bash
cd ~/fan && source ~/fanenv/bin/activate
python benchmark_latency.py models/echolab/model.tflite \
       --threads 4 --runs 200 --json pi_bench.json         # numbers for the report
python pi_inference.py --device 1                          # live
python pi_inference.py --simulate dataset/fan/test/abnormal         # demo without a fan / mic
```

Open `http://<pi-ip>:8080/` from a phone on the same network for the status page.
Logs are written to `logs/events_YYYYMMDD.csv`.

**Calibrating `--min-rms`:** run with the fan **off**, note the `rms=` values printed, and set
`--min-rms` to about 2× that value. Below it the system reports `NO_SIGNAL` and does not classify.

**Re-tuning on site:** thresholds are in z-units of healthy sound (`models/echolab/config.json`). Record your fan running
normally, run `python pi_inference.py --simulate normal_recordings/` and set `--threshold` slightly above the largest z you
see, or raise `--k` (windows needed for an alarm). On the PC, the website's *Calibrate on my own fan* does this automatically.

## 7. Run at boot (systemd)

```bash
sudo cp fan-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now fan-monitor
journalctl -u fan-monitor -f
```

## 8. Validation checklist for the report (do these ON the Pi)

1. `benchmark_latency.py` → model size, latency (median / p95), CPU temperature, throttle flag
   (`throttled=0x0` means no throttling).
2. 24 h soak test with the fan running normally → count ALARM events = measured false-alarm rate.
3. Controlled fault: tape a small weight on one blade / loosen a mount (only if safe and permitted)
   or play recorded abnormal clips next to the mic → detection delay.
4. Power cut test: the service must restart cleanly (`systemctl status fan-monitor`).
5. Memory: `htop` or `ps -o rss -p $(pgrep -f pi_inference)`.

Safety: do not introduce faults into a production fan; use a spare fan on a bench.
