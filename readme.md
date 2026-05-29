# GMM-HMM ASR kaldi+commonvoice pipeline (Tubitak 2209-A)

Our Tubitak 2209-A supported project was about GMM-HMM ASR pipeline to work on microcontrollers. There has been several attempts to learn and do it. Finally we did make some progress on learning it and gained enough information to start with kaldi library. This repository has both my failed attempts (not all of them) and my current pipeline with kaldi library. 

- **Note**: Repository has 3 deprecated readmes. I am gonna merge them into this one.

What i did in this pipeline is created a custom dockerfile instead of kaldi's default to not implement every step of library installations every time when i start kaldi. Also did the same thing and automatized the python and training scripts in docker_baslat.sh file. You can find these file's explanations in kaldi/docs.

- dockerfile notes: [kaldi/docs/dockerfile_doc_readme.md](kaldi/docs/dockerfile_doc_readme.md)
- docker_baslat.sh notes: [kaldi/docs/sh_doc_readme.md](kaldi/docs/sh_doc_readme.md)

- Build image (run once):
```
docker build -t kaldi-turkce project-scripts-path
```

- Run training pipeline (each time):
```
docker run -it \
  -v cv-dataset-path:/data/commonvoice \
  -v project-scripts-path:/data/proje \
  kaldi-turkce
```

## Current progress:

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
