# GMM-HMM ASR kaldi+commonvoice pipeline (Tubitak 2209-A)

Our Tubitak 2209-A supported project was about GMM-HMM ASR pipeline to work on microcontrollers. There has been several attempts to learn and do it. Finally we did make some progress on learning it and gained enough information to start with kaldi library. This repository has both my failed attempts (not all of them) and my current pipeline with kaldi library. 

- Repository has 3 deprecated readmes. I am gonna merge them to make one in the future.

What i did in this pipeline is created a custom dockerfile instead of kaldi's default to not implement every step of library installations every time when i start kaldi. Also did the same thing and automatized the python and training scripts in docker_baslat.sh file.

## Current progress:

- DENEY 1 - Temel sistem
    Veri    : 1000 train, 100 dev, 100 test (train.tsv)
    LM      : küçük (1000 cümle)
    Model   : Monophone
    Sonuç   : %96.62 WER

- DENEY 2 - Veri artışı
    Veri    : 10000 train, 500 dev, 500 test (train.tsv)
    LM      : küçük (10000 cümle)
    Model   : Monophone
    Sonuç   : %89.94 WER

- DENEY 3 - Model karmaşıklığı artışı
    Veri    : 10000 train, 500 dev, 500 test (train.tsv)
    LM      : küçük (10000 cümle)
    Model   : Monophone → Triphone → LDA+MLLT → SAT
    Sonuç   : Mono %89.94 / Tri1 %80.79 / Tri2 %79.91 / SAT %79.78

- DENEY 4 - Veri + model artışı (LATEST)
    Veri    : 40000 train, 1000 dev, 1000 test (train.tsv)
    LM      : küçük (40000 cümle)
    Gaussian: 2500/15000
    Model   : Mono → Tri1 → Tri2 → SAT
    Sonuç   : Mono %86.69 / Tri1 %71.39 / Tri2 %67.61 / SAT %65.41

```
- DENEY 5 - Büyük LM + büyük Gaussian (REVERTED - data leakage suspicion on language model)
    Veri    : 40000 train, 1000 dev, 1000 test (train.tsv)
    LM      : büyük (40000 + 120000 validated cümle (trainx2 ve dev ve test çıkarılmamış halde))
    Gaussian: 4000/40000
    Model   : Mono → Tri1 → Tri2 → SAT
    Sonuç   : Mono %80.80 / Tri1 %63.88 / Tri2 %59.69 / SAT %56.52

- DENEY 6 - Validated.tsv bölünmüş (REVERTED - data leakage suspicition on Acoustic and language model)
    Veri    : ~105000 train, ~5500 dev, ~5500 test (instead of original train-test-dev. i splitted the validated.tsv into parts with the suspect of data leakage)
    LM      : büyük
    Gaussian: 4000/40000
    Model   : Mono → Tri1 → Tri2 → SAT
    Sonuç   : SAT %14.44 WER ( PARTIAL )  ← data leakage şüphesi
    Not     : ( PARTIAL ) = lexicon dışı kelimeler decode edilemedi
                WER güvenilir değil

- DENEY 7 - Devasa LM + büyük Gaussian + Normal AM (REVERTED - pre processing incompatibility)
    Veri    : 40000 train, ~3000 dev, ~3000 test (orijinal Mozilla cv)
    LM      : ~385k validated_sentences (dev ve test cümleleri çıkarılmış)
    Gaussian: 4000/40000
    Model   : Mono → Tri1 → Tri2 → SAT
    Sonuç   : ?
```