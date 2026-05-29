# Kaldi Türkçe Konuşma Tanıma - Kurulum ve Kullanım

## Proje Yapısı

```
27-04-26-new-attempt/
    Dockerfile                  ← Özel Docker image tanımı
    docker_baslat.sh            ← Docker başlangıç scripti (her şeyi otomatik çalıştırır)
    kaldi_data_prep.py          ← Veri hazırlama scripti
    kaldi_lexicon_olustur.py    ← Lexicon oluşturma scripti
    kaldi_dict_hazirla.py       ← Sözlük dosyaları oluşturma scripti

cv-corpus-24.0-2025-12-05-tr/cv-corpus-24.0-2025-12-05/tr/
    clips/                      ← MP3 ses dosyaları
    train.tsv                   ← Eğitim verisi
    dev.tsv                     ← Doğrulama verisi
    test.tsv                    ← Test verisi
```

---

## Image Build Etme (Bir Kez Yapılır)

KenLM dahil her şeyi kurar, 10-15 dakika sürer:

```bash
docker build -t kaldi-turkce /home/turkerkiv/Desktop/software-projects/tubitak/27-04-26-new-attempt/
```

---

## Docker Başlatma

```bash
docker run -it \
  -v /home/turkerkiv/Desktop/software-projects/tubitak/cv-corpus-24.0-2025-12-05-tr/cv-corpus-24.0-2025-12-05/tr:/data/commonvoice \
  -v /home/turkerkiv/Desktop/software-projects/tubitak/27-04-26-new-attempt:/data/proje \
  kaldi-turkce
```

Docker başlayınca docker_baslat.sh otomatik çalışır ve şu adımları sırayla yapar:

```
1.  Veri hazırla          (kaldi_data_prep.py)
2.  Dosyaları sırala
3.  MFCC çıkar            (make_mfcc.sh)
4.  CMVN hesapla          (compute_cmvn_stats.sh)
5.  Lexicon oluştur       (kaldi_lexicon_olustur.py)
6.  Sözlük dosyaları      (kaldi_dict_hazirla.py)
7.  Lang klasörü          (prepare_lang.sh)
8.  N-gram dil modeli     (lmplz + format_lm.sh)
9.  Monophone GMM-HMM     (train_mono.sh)
10. Decode graph          (mkgraph.sh)
11. Test decode           (decode.sh)
12. WER sonucu göster
```

Her şey bitince bash'e düşer, WER sonucunu görürsün.

---

## Veri Miktarını Ayarlama

`kaldi_data_prep.py` içinde:

```python
MAX_TRAIN = 10000   ← eğitim örneği
MAX_DEV   = 500     ← doğrulama örneği
MAX_TEST  = 500     ← test örneği
```

---

## Oluşturulan Dosyalar

```
/data/kaldi_tr/
    tr_train/           ← eğitim verisi (text, wav.scp, utt2spk, spk2utt)
    tr_dev/             ← doğrulama verisi
    tr_test/            ← test verisi
    mfcc/               ← MFCC özellikleri
    local/
        dict/
            lexicon.txt             ← kelime - fonem eşleşmesi
            silence_phones.txt      ← SIL, SPN
            nonsilence_phones.txt   ← 27 ASCII fonem
            optional_silence.txt    ← SIL
        lm_train.txt    ← dil modeli eğitim metni
        lm.arpa         ← trigram dil modeli (ARPA format)
        lm.arpa.gz      ← sıkıştırılmış ARPA
    lang/               ← Kaldi dil klasörü (L.fst)
    lang_test/          ← Dil modeli dahil (G.fst)
    exp/
        mono/           ← Monophone GMM-HMM modeli
        mono/graph/     ← Decode graph (HCLG.fst)
        mono/decode_test/ ← Decode sonuçları
```

---

## Fonem Seti (27 fonem)

```
a  b  c  d  e  f  g  h  i  ii
j  k  l  m  n  o  oe p  r  s
sh t  u  ue v  z  zh
```

IPA - ASCII dönüşümleri:
```
ɫ → l     ʒ → zh    ɛ → e
ɯ → ii    ʃ → sh    ɾ → r
ɟ → j     t̪ → t     d̪ → d
n̪ → n     s̪ → s     ʎ → ly
œ → oe    ɔ → o     ɡ → g
ɪ → i     ʊ → u     ː → (sil)
```

---

## WER Sonucuna Bakma

Docker içinde:

```bash
cat /data/kaldi_tr/exp/mono/decode_test/scoring_kaldi/best_wer
```

---

## Sıradaki Adımlar

```
✅ Monophone GMM-HMM + Trigram N-gram sistemi çalışıyor

[ ] Veriyi artır (MAX_TRAIN = 10000)
[ ] Triphone GMM-HMM eğit
[ ] SAT (Speaker Adaptive Training) ekle
[ ] WER karşılaştır
```

---

## Notlar

- Image bir kez build edilir, tekrar build gerekmez
- /data/kaldi_tr Docker kapanınca silinir, her başlatışta yeniden oluşturulur
- NJ=8 olarak ayarlı (8 paralel iş), makinede 16 çekirdek var


DENEY 1 - Temel sistem
  Veri    : 1000 train, 100 dev, 100 test (train.tsv)
  LM      : küçük (1000 cümle)
  Model   : Monophone
  Sonuç   : %96.62 WER
DENEY 2 - Veri artışı
  Veri    : 10000 train, 500 dev, 500 test (train.tsv)
  LM      : küçük (10000 cümle)
  Model   : Monophone
  Sonuç   : %89.94 WER
DENEY 3 - Model karmaşıklığı artışı
  Veri    : 10000 train, 500 dev, 500 test (train.tsv)
  LM      : küçük (10000 cümle)
  Model   : Monophone → Triphone → LDA+MLLT → SAT
  Sonuç   : Mono %89.94 / Tri1 %80.79 / Tri2 %79.91 / SAT %79.78
DENEY 4 - Veri + model artışı
  Veri    : 40000 train, 1000 dev, 1000 test (train.tsv)
  LM      : küçük (40000 cümle)
  Gaussian: 2500/15000
  Model   : Mono → Tri1 → Tri2 → SAT
  Sonuç   : Mono %86.69 / Tri1 %71.39 / Tri2 %67.61 / SAT %65.41 <- EN İYİ DOĞRU OLAN
DENEY 5 - Büyük LM + büyük Gaussian
  Veri    : 40000 train, 1000 dev, 1000 test (train.tsv)
  LM      : büyük (40000 + 120000 validated cümle (trainx2 ve dev ve test çıkarılmamış halde))
  Gaussian: 4000/40000
  Model   : Mono → Tri1 → Tri2 → SAT
  Sonuç   : Mono %80.80 / Tri1 %63.88 / Tri2 %59.69 / SAT %56.52 ← EN İYİ
DENEY 6 - Validated.tsv bölünmüş (SORUNLU)
  Veri    : ~105000 train, ~5500 dev, ~5500 test (validated.tsv bölünmüş)
  LM      : büyük
  Gaussian: 4000/40000
  Model   : Mono → Tri1 → Tri2 → SAT
  Sonuç   : SAT %14.44 WER [PARTIAL] ← data leakage şüphesi
  Not     : [PARTIAL] = lexicon dışı kelimeler decode edilemedi
             WER güvenilir değil
DENEY 7 - Devasa LM + büyük Gaussian + Normal AM
  Veri    : 40000 train, ~3000 dev, ~3000 test (orijinal Mozilla bölmesi)
  LM      : ~385k validated_sentences (dev ve test cümleleri çıkarılmış)
  Gaussian: 4000/40000
  Model   : Mono → Tri1 → Tri2 → SAT
  Sonuç   : ?