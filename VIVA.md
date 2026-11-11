# Viva preparation (EchoLab deck protocol)

Answer from your own measured numbers (`models/echolab/metrics.json`, the website's *Model & Results* tab), not from this file.

**1. What problem does the project solve?** Fans in factories and labs run all day; worn bearings or blocked air go unnoticed until
overheating and downtime. A microphone (about Rs. 150) can listen continuously where a contact vibration sensor costs about Rs. 15k.

**2. Why only healthy sounds for training?** In practice you have hours of a healthy fan and almost no recordings of faults.
The model must learn what *normal* sounds like; the 341 faulty clips are held out and used only to test.

**3. How does a classifier learn without faulty examples?** Each batch is 50 % healthy and 50 % *pseudo-faulty* clips made from
real healthy ones by injecting fault patterns (4-8 kHz rattle, clicks, tonal lines, amplitude modulation, raised noise). It learns to give unusual
sound a high score. Honest limit: injected patterns may not look like real faults; the unseen real test measures the transfer.

**4. Why full 10 s and a spectrogram?** The baseline listens to 0.3 s. Faults such as intermittent rattle or slow amplitude modulation
need seconds. The log-Mel spectrogram (128 x 313) shows faults as bright high-pitched lines, which an image network can read.
313 = 1 + 160000/512 (frame 1024, hop 512, centre-padded).

**5. Why MobileNetV2 and transfer learning?** Small and fast (depthwise-separable convolutions), and ImageNet filters (edges, textures)
transfer to spectrograms, so about 1000 healthy clips are enough. Phase 1 trains only the head; phase 2 unfreezes the last layers with
BatchNorm frozen.

**6. Where does PCA come in, and why?** (a) PCA on the spectrogram's 5-frame context vectors and (b) PCA on the network's 1280-d embedding,
both fitted on healthy sound only. A sound the healthy subspace cannot reconstruct has a large reconstruction error. They need no faulty
clip and are independent of the pseudo-fault assumption, so they complement the CNN. The PCA Lab tab shows the scree curve and a map.

**7. How are the three scores combined?** Each is converted to a z-score against the out-of-fold *healthy* distribution (median/MAD, logit
for the CNN, log for PCA errors) and averaged with equal weights. The weights were fixed in advance; choosing the best-looking
combination on test data would be test-set tuning.

**8. Explain stratified 5-fold CV here. Why not leave-one-out, 2-fold, time-series, simple K-fold?** 1003 healthy clips → 5 folds of
about 802 train / 201 validation, every fold with the same 1:1 healthy : pseudo-fault mix. LOOCV would need about 1000 trainings and
its training sets overlap almost completely; 2-fold trains on half the data (pessimistic); clips are independent so time-series CV is
unnecessary; plain K-fold can upset the class mix. Result = average AUC ± spread over folds.

**9. Bias and variance?** Bias: model too simple → errors on training and new data. Variance: too sensitive to the training clips →
fails on new data. Read them from the validation level, the train–validation gap and the spread across folds (Model & Results).

**10. How is the threshold chosen without cheating?** Two rules. *Label-free:* 95th percentile of out-of-fold healthy scores (needs no
faulty clip, about 5 % false alarms). *ROC-best:* Youden-optimal, **cross-fitted**: chosen on 4/5 of the test clips and applied to the
remaining 1/5, so no clip is scored with a threshold tuned on itself. The reported accuracy/F1 use these.

**11. Your accuracy is X %. Is that good?** Test set = 341 faulty + 100 healthy, so "always faulty" scores 77.3 %. Quote balanced
accuracy, AUC, pAUC and F1 next to accuracy.

**12. What are AUC and pAUC?** AUC = probability that a random faulty clip scores higher than a random healthy one (threshold-free).
pAUC = the area only for false-positive rate ≤ 0.1 (DCASE metric); it rewards finding faults while keeping false alarms low.

**13. What does Grad-CAM show?** The regions of the spectrogram that most increase the fault score. Caution: the injected training patterns are
partly placed at 4-8 kHz, so attention there is expected and is not independent proof about real faults.

**14. Quantisation?** Dynamic-range quantisation stores weights as int8: smaller (measured size is on the Results tab) and faster on a CPU.
The exported model has two outputs (fault score and embedding); the website checks the TFLite model against Keras on test clips.

**15. Why NumPy features?** The identical code runs in training, the website and the Raspberry Pi: no train/serve skew, and no librosa on
the Pi. It is tested against librosa to about 1e-5 dB.

**16. Which deck targets were met?** Read the target table (Model & Results, `report/REPORT.md` §7.5): AUC 90 %+, pAUC about 78 %,
F1 0.87, speed. They are MET or NOT MET from the data. State the result either way.

**17. Is the baseline comparison fair?** The baseline is unsupervised (healthy only), and so is this project's training. The ROC-best
threshold uses test labels (cross-fitted); the label-free threshold does not. Also verify the baseline number for fan id_00 in the DCASE
2020 Task 2 results.

**18. Will it work on a different fan or microphone?** Not guaranteed. Use *Calibrate on my own fan* (≥ 30 s of normal sound): scores are
re-centred. Test on a real lab fan as future work.

**19. Limitations?** Pseudo-faults are injected patterns, not real faults; only id_00; ROC-best threshold needs some labelled faults; Pi speed must be measured on
a Pi; a screening signal, not a diagnosis.

**20. Future work and what is already done?** Lab fan with a mic (record in the website), more faults (extend `FAULT_KINDS`), Telegram alerts
(implemented), phone access (QR), pumps and valves (`--machine pump`).
