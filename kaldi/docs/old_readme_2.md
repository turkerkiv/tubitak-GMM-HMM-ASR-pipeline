# Kaldi Türkçe Konuşma Tanıma - Kurulum ve Kullanım

## Proje Yapısı

```
27-04-26-new-attempt/
    Dockerfile                  ← Özel Docker image tanımı
    docker_baslat.sh            ← Docker başlangıç scripti
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

## Dockerfile İçeriği

```dockerfile
FROM kaldiasr/kaldi

RUN apt-get update && apt-get install -y \
    sox \
    libsox-fmt-all \
    python3-pip \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

RUN pip install phonemizer --break-system-packages

WORKDIR /opt/kaldi/egs/commonvoice/s5

ENTRYPOINT ["bash", "/data/proje/docker_baslat.sh"]
```

---

## docker_baslat.sh İçeriği

Bu script Docker her başlatıldığında otomatik çalışır:

```bash
#!/bin/bash

echo "=== Kaldi Türkçe ASR Başlatılıyor ==="

cd /opt/kaldi/egs/commonvoice/s5
source ./path.sh

echo "1. Veri hazırlanıyor..."
python3 /data/proje/kaldi_data_prep.py

echo "2. Dosyalar sıralanıyor..."
for dir in /data/kaldi_tr/tr_train /data/kaldi_tr/tr_dev /data/kaldi_tr/tr_test; do
  sort $dir/utt2spk -o $dir/utt2spk
  sort $dir/text -o $dir/text
  sort $dir/wav.scp -o $dir/wav.scp
done

echo "3. MFCC çıkarılıyor..."
steps/make_mfcc.sh --nj 4 --cmd "run.pl" /data/kaldi_tr/tr_train /data/kaldi_tr/log/train /data/kaldi_tr/mfcc/train
steps/make_mfcc.sh --nj 4 --cmd "run.pl" /data/kaldi_tr/tr_dev /data/kaldi_tr/log/dev /data/kaldi_tr/mfcc/dev
steps/make_mfcc.sh --nj 4 --cmd "run.pl" /data/kaldi_tr/tr_test /data/kaldi_tr/log/test /data/kaldi_tr/mfcc/test

echo "4. CMVN hesaplanıyor..."
steps/compute_cmvn_stats.sh /data/kaldi_tr/tr_train /data/kaldi_tr/log/cmvn_train /data/kaldi_tr/mfcc/train
steps/compute_cmvn_stats.sh /data/kaldi_tr/tr_dev /data/kaldi_tr/log/cmvn_dev /data/kaldi_tr/mfcc/dev
steps/compute_cmvn_stats.sh /data/kaldi_tr/tr_test /data/kaldi_tr/log/cmvn_test /data/kaldi_tr/mfcc/test

echo "5. Lexicon oluşturuluyor..."
python3 /data/proje/kaldi_lexicon_olustur.py
python3 /data/proje/kaldi_dict_hazirla.py

echo "=== Hazır! ==="
exec bash
```

---

## Image Build Etme (Bir Kez Yapılır)

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

Docker başlayınca docker_baslat.sh otomatik çalışır ve her şeyi hazırlar.

---

## Oluşturulan Dosyalar

```
/data/kaldi_tr/
    tr_train/
        text        ← utterance_id transkript
        wav.scp     ← utterance_id ses_dosyası_yolu
        utt2spk     ← utterance_id speaker_id
        spk2utt     ← speaker_id utterance_id_listesi
    tr_dev/         (aynı yapı)
    tr_test/        (aynı yapı)
    mfcc/
        train/      ← MFCC özellikleri
        dev/
        test/
    local/
        dict/
            lexicon.txt             ← kelime - fonem eşleşmesi
            silence_phones.txt      ← SIL, SPN
            nonsilence_phones.txt   ← 27 ASCII fonem
            optional_silence.txt    ← SIL
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

## Sıradaki Adımlar

```
✅ Veri hazırlama (kaldi_data_prep.py)
✅ MFCC çıkarma (make_mfcc.sh)
✅ CMVN normalizasyonu (compute_cmvn_stats.sh)
✅ Lexicon oluşturma (kaldi_lexicon_olustur.py)
✅ Sözlük dosyaları (kaldi_dict_hazirla.py)

[ ] Lang klasörü hazırlama (prepare_lang.sh)
[ ] N-gram dil modeli eğitimi
[ ] Monophone GMM-HMM eğitimi
[ ] Triphone GMM-HMM eğitimi
[ ] Decode et, WER hesapla
```

---

## Notlar

- Docker her kapandığında kurulumlar sıfırlanır ama kaldi-turkce image kalır, tekrar build gerekmez
- /data/kaldi_tr klasörü volume olmadığı için Docker kapanınca silinir, her başlatışta yeniden oluşturulur
- Daha fazla veri için kaldi_data_prep.py içindeki MAX_TRAIN değerini artır (şu an 1000)
