# GMM-HMM ASR kaldi+commonvoice pipeline (Tubitak 2209-A)

Our Tubitak 2209-A supported project was about GMM-HMM ASR pipeline to work on microcontrollers. There has been several attempts to learn and do it. Finally we did make some progress on learning it and gained enough information to start with kaldi library. This repository has both my failed attempts (not all of them) and my current pipeline with kaldi library. 

- **Note**: Repository has 3 deprecated readmes. I am gonna merge them into this one.

What i did in this pipeline is created a custom dockerfile instead of kaldi's default to not implement every step of library installations every time when i start kaldi. Also did the same thing and automatized the python and training scripts in docker_baslat.sh file. You can find these file's explanations in kaldi/docs.

- Dockerfile notes: [kaldi/docs/dockerfile_doc_readme.md](kaldi/docs/dockerfile_doc_readme.md)
- docker_baslat.sh notes: [kaldi/docs/sh_doc_readme.md](kaldi/docs/sh_doc_readme.md)

### Build image

Run this once:
```
docker build -t kaldi-turkce project-scripts-path
```

### Run training pipeline

Run this each time:
```
docker run -it \
  -v cv-dataset-path:/data/commonvoice \
  -v project-scripts-path:/data/proje \
  kaldi-turkce

docker run -it --gpus all \
-v /home/turkerkiv/Desktop/software-projects/tubitak/dataset-cv-24.0:/data/commonvoice \
-v /home/turkerkiv/Desktop/software-projects/tubitak/kaldi:/data/proje \
kaldi-turkce-gpu-2

```

## Current progress:

These numbers are the ones I kept in the final report.

### EXPERIMENT 1 - Base system

- Data: 1000 train, 100 dev, 100 test (train.tsv)
- LM: small (1000 sentences)
- Model: Monophone
- Result: %96.62 WER

### EXPERIMENT 2 - Data increase

- Data: 10000 train, 500 dev, 500 test (train.tsv)
- LM: small (10000 sentences)
- Model: Monophone
- Result: %89.94 WER

### EXPERIMENT 3 - Model complexity increase

- Data: 10000 train, 500 dev, 500 test (train.tsv)
- LM: small (10000 sentences)
- Model: Monophone -> Triphone -> LDA+MLLT -> SAT
- Result: Mono %89.94 / Tri1 %80.79 / Tri2 %79.91 / SAT %79.78

### EXPERIMENT 4 - Data + model increase

- Data: 40000 train, 1000 dev, 1000 test (train.tsv)
- LM: small (40000 sentences)
- Gaussian: 2500/15000
- Model: Mono -> Tri1 -> Tri2 -> SAT
- Result: Mono %86.69 / Tri1 %71.39 / Tri2 %67.61 / SAT %65.41

### EXPERIMENT 5 - Larger Gaussian (LATEST)
- Data: 40000 train, 3000 dev, 3000 test (train.tsv)
- LM: small (40000 sentences)
- Gaussian: 4000/40000 (change only on SAT)
- Model: Mono -> Tri1 -> Tri2 -> SAT
- Result: Mono %83.63 / Tri1 %67.11 / Tri2 %62.59 / SAT %56.17

### EXPERIMENT 6 - TDNN integration

- Data: 40000 train, 3000 dev, 3000 test (train.tsv)
- LM: small (40000 sentences)
- Model: SAT alignments -> TDNN
- TDNN: 5 layers, dim=512, Kaldi nnet3
- Result: TDNN %54.71 WER

---
<details>
<summary>Deprecated experiments (reverted)</summary>

### EXPERIMENT 5 - Large LM + large Gaussian (REVERTED - data leakage suspicion on language model)

- Data: 40000 train, 1000 dev, 1000 test (train.tsv)
- LM: large (40000 + 120000 validated sentences (trainx2 and dev/test not removed))
- Gaussian: 4000/40000
- Model: Mono -> Tri1 -> Tri2 -> SAT
- Result: Mono %80.80 / Tri1 %63.88 / Tri2 %59.69 / SAT %56.52

### EXPERIMENT 6 - Validated.tsv split (REVERTED - data leakage suspicition on Acoustic and language model)

- Data: ~105000 train, ~5500 dev, ~5500 test (instead of original train-test-dev. i splitted the validated.tsv into parts with the suspect of data leakage)
- LM: large
- Gaussian: 4000/40000
- Model: Mono -> Tri1 -> Tri2 -> SAT
- Result: SAT %14.44 WER (PARTIAL) <- data leakage suspicion
- Note: (PARTIAL) = out-of-lexicon words could not be decoded; WER not reliable

### EXPERIMENT 7 - Huge LM + large Gaussian + Normal AM (REVERTED - pre processing incompatibility)

- Data: 40000 train, ~3000 dev, ~3000 test (original Mozilla CV)
- LM: ~385k validated_sentences (dev and test sentences removed)
- Gaussian: 4000/40000
- Model: Mono -> Tri1 -> Tri2 -> SAT
- Result: ?
</details>

## Final report add-ons

I checked the final pdf too, so I added the extra things that were missing from the previous readme.

### Final report summary

- Project title in the report: HMM-N-gram hybrid model with high efficiency speech recognition system
- Project dates: 09/04/2025 - 08/07/2026
- Final best GMM-HMM result: SAT %56.17 WER
- Final TDNN result: %54.71 WER
- TDNN setup: 5 layers, dim=512, trained on SAT alignments with Kaldi nnet3
- The system was built on Kaldi + Docker + Common Voice v24 Turkish data

### Raspberry Pi 4 test

- I also tested the models on Raspberry Pi 4 Model B (Cortex-A72, 1.5 GHz, 4 core, 4 GB RAM)
- Test size: 1000 samples
- Mono: %83.48 WER, 560 sn, RTF 0.13, 260 MB RAM
- Tri1: %67.16 WER, 1120 sn, RTF 0.26, 274 MB RAM
- Tri2: %63.06 WER, 1550 sn, RTF 0.36, 281 MB RAM
- Tri3 (SAT): %56.30 WER, 2410 sn, RTF 0.56, 315 MB RAM
- TDNN: %54.15 WER, 7168 sn, RTF 1.665, 1336 MB RAM
- For low resource and real-time usage, I preferred Tri3 because its RTF stayed below 1

### Output notes

- I produced a fully automated Kaldi training pipeline
- The outputs include Monophone, Tri1, Tri2, Tri3 (SAT), and TDNN models
- I also prepared a KenLM trigram language model and a Turkish phoneme lexicon with about 44,346 words and 28 ASCII phonemes

### What I would keep in mind from the final report

- I did not hit the original %90 accuracy target, but I still got a clear improvement from %96.62 WER to %54.71 WER
- For future work, I would go for i-vector speaker adaptation, speed perturbation, and LF-MMI chain models
