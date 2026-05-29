# Kaldi Türkçe Konuşma Tanıma - Kurulum ve Kullanım

## Proje Yapısı

```
27-04-26-new-attempt/
    kaldi_data_prep.py       ← Veri hazırlama scripti
    
cv-corpus-24.0-2025-12-05-tr/cv-corpus-24.0-2025-12-05/tr/
    clips/                   ← MP3 ses dosyaları
    train.tsv                ← Eğitim verisi
    dev.tsv                  ← Doğrulama verisi
    test.tsv                 ← Test verisi
```

---

## 1. Docker Başlatma

Her seferinde Docker'ı şu komutla başlat:

```bash
docker run -it \
  -v /home/turkerkiv/Desktop/software-projects/tubitak/cv-corpus-24.0-2025-12-05-tr/cv-corpus-24.0-2025-12-05/tr:/data/commonvoice \
  -v /home/turkerkiv/Desktop/software-projects/tubitak/27-04-26-new-attempt:/data/proje \
  kaldiasr/kaldi bash
```

Bu komut:
- Common Voice Türkçe verisini `/data/commonvoice` olarak bağlar
- Proje klasörünü `/data/proje` olarak bağlar
- Kaldi kurulu Docker container'ı başlatır

---

## 2. MP3 Desteği Kurulumu

Docker her başlatıldığında bu adımı tekrarla (kurulum sıfırlanıyor):

```bash
apt-get update && apt-get install -y sox libsox-fmt-all
```

---

## 3. Kaldi Dizinine Geç

```bash
cd /opt/kaldi/egs/commonvoice/s5
. ./path.sh
```

`path.sh` Kaldi araçlarını PATH'e ekler, her Docker başlatışında çalıştır.

---

## 4. Veri Hazırlama

Docker dışında sadece bir kez yapılır, `/data/kaldi_tr` klasörü volume'da kalır:

```bash
python3 /data/proje/kaldi_data_prep.py
```

Bu script şunları oluşturur:
```
/data/kaldi_tr/
    tr_train/
        text      ← utterance_id transkript
        wav.scp   ← utterance_id ses_dosyası_yolu
        utt2spk   ← utterance_id speaker_id
        spk2utt   ← speaker_id utterance_id_listesi
    tr_dev/
        (aynı yapı)
    tr_test/
        (aynı yapı)
```

Dosyaları sırala (bir kez yapılır):

```bash
for dir in /data/kaldi_tr/tr_train /data/kaldi_tr/tr_dev /data/kaldi_tr/tr_test; do
  sort $dir/utt2spk -o $dir/utt2spk
  sort $dir/text -o $dir/text
  sort $dir/wav.scp -o $dir/wav.scp
done
```

Veriyi doğrula:

```bash
utils/validate_data_dir.sh --no-feats /data/kaldi_tr/tr_train 2>&1 | grep -v "non-printable"
utils/validate_data_dir.sh --no-feats /data/kaldi_tr/tr_dev 2>&1 | grep -v "non-printable"
utils/validate_data_dir.sh --no-feats /data/kaldi_tr/tr_test 2>&1 | grep -v "non-printable"
```

Çıktı yoksa veri hazır demektir.

---

## 5. MFCC Çıkarma

```bash
steps/make_mfcc.sh --nj 4 --cmd "run.pl" \
  /data/kaldi_tr/tr_train \
  /data/kaldi_tr/log/train \
  /data/kaldi_tr/mfcc/train
```

Bu komut:
- `--nj 4` → 4 paralel iş çalıştır
- `--cmd "run.pl"` → komutları run.pl ile çalıştır
- Tüm ses dosyalarından MFCC özniteliklerini çıkarır
- Sonuçları `/data/kaldi_tr/mfcc/train` klasörüne yazar

Başarılı çıktı:
```
steps/make_mfcc.sh: Succeeded creating MFCC features for tr_train
```

---

## 6. Sıradaki Adımlar (Henüz Yapılmadı)

```
[ ] Dev ve test için MFCC çıkar
[ ] CMVN normalizasyonu yap
[ ] Sözlük (lexicon) oluştur
[ ] Dil modeli (N-gram) eğit
[ ] Monophone GMM-HMM eğit
[ ] Triphone GMM-HMM eğit
[ ] Decode et, WER hesapla
```

---

## Notlar

- Docker her kapandığında apt kurulumları sıfırlanır, 2. adımı tekrarla
- `/data/kaldi_tr` klasörü volume'da olduğu için veri kaybolmaz
- `kaldi_data_prep.py` şu an 1000 train, 100 dev, 100 test örnek kullanıyor
- Daha fazla veri için scriptteki `MAX_TRAIN` değerini artır


## henüz düzenlenmemiş sonradan yaptıklarım

steps/make_mfcc.sh --nj 4 --cmd "run.pl" \
  /data/kaldi_tr/tr_dev \
  /data/kaldi_tr/log/dev \
  /data/kaldi_tr/mfcc/dev

steps/make_mfcc.sh --nj 4 --cmd "run.pl" \
  /data/kaldi_tr/tr_test \
  /data/kaldi_tr/log/test \
  /data/kaldi_tr/mfcc/test

---

steps/compute_cmvn_stats.sh \
  /data/kaldi_tr/tr_train \
  /data/kaldi_tr/log/cmvn_train \
  /data/kaldi_tr/mfcc/train

steps/compute_cmvn_stats.sh \
  /data/kaldi_tr/tr_dev \
  /data/kaldi_tr/log/cmvn_dev \
  /data/kaldi_tr/mfcc/dev

steps/compute_cmvn_stats.sh \
  /data/kaldi_tr/tr_test \
  /data/kaldi_tr/log/cmvn_test \
  /data/kaldi_tr/mfcc/test

---


# hızlı her açılışta tekrarlanacak kod listesi

```
0. docker run -it   -v /home/turkerkiv/Desktop/software-projects/tubitak/cv-corpus-24.0-2025-12-05-tr/cv-corpus-24.0-2025-12-05/tr:/data/commonvoice   -v /home/turkerkiv/Desktop/software-projects/tubitak/27-04-26-new-attempt:/data/proje   kaldiasr/kaldi bash

1. apt-get update && apt-get install -y sox libsox-fmt-all python3-pip espeak-ng

2. cd /opt/kaldi/egs/commonvoice/s5

3. . ./path.sh

4. python3 /data/proje/kaldi_data_prep.py

5. for dir in /data/kaldi_tr/tr_train /data/kaldi_tr/tr_dev /data/kaldi_tr/tr_test; do
     sort $dir/utt2spk -o $dir/utt2spk
     sort $dir/text -o $dir/text
     sort $dir/wav.scp -o $dir/wav.scp
   done

6. steps/make_mfcc.sh --nj 4 --cmd "run.pl" /data/kaldi_tr/tr_train /data/kaldi_tr/log/train /data/kaldi_tr/mfcc/train
   steps/make_mfcc.sh --nj 4 --cmd "run.pl" /data/kaldi_tr/tr_dev /data/kaldi_tr/log/dev /data/kaldi_tr/mfcc/dev
   steps/make_mfcc.sh --nj 4 --cmd "run.pl" /data/kaldi_tr/tr_test /data/kaldi_tr/log/test /data/kaldi_tr/mfcc/test

7. steps/compute_cmvn_stats.sh /data/kaldi_tr/tr_train /data/kaldi_tr/log/cmvn_train /data/kaldi_tr/mfcc/train
   steps/compute_cmvn_stats.sh /data/kaldi_tr/tr_dev /data/kaldi_tr/log/cmvn_dev /data/kaldi_tr/mfcc/dev
   steps/compute_cmvn_stats.sh /data/kaldi_tr/tr_test /data/kaldi_tr/log/cmvn_test /data/kaldi_tr/mfcc/test

8. pip install phonemizer --break-system-packages

9. python3 /data/proje/kaldi_lexicon_olustur.py

```

## her defasında tekrar baştan yazmamak için Dockerfile oluşturduk

```
FROM kaldiasr/kaldi

# Sistem paketleri
RUN apt-get update && apt-get install -y \
    sox \
    libsox-fmt-all \
    python3-pip \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Python paketleri
RUN pip install phonemizer --break-system-packages

# Varsayılan çalışma dizini
WORKDIR /opt/kaldi/egs/commonvoice/s5
```

- sonra:
sadece 1 kere build et bunla
```
docker build -t kaldi-turkce /home/turkerkiv/Desktop/software-projects/tubitak/27-04-26-new-attempt/
```

- docker artık şunla başlar hep:
```
docker run -it \
  -v /home/turkerkiv/Desktop/software-projects/tubitak/cv-corpus-24.0-2025-12-05-tr/cv-corpus-24.0-2025-12-05/tr:/data/commonvoice \
  -v /home/turkerkiv/Desktop/software-projects/tubitak/27-04-26-new-attempt:/data/proje \
  kaldi-turkce bash
```

- artık sadece kod özelindeki şeyler kalır geriye:
```
1. . ./path.sh

2. python3 /data/proje/kaldi_data_prep.py

3. for dir in /data/kaldi_tr/tr_train /data/kaldi_tr/tr_dev /data/kaldi_tr/tr_test; do
     sort $dir/utt2spk -o $dir/utt2spk
     sort $dir/text -o $dir/text
     sort $dir/wav.scp -o $dir/wav.scp
   done

4. steps/make_mfcc.sh --nj 4 --cmd "run.pl" /data/kaldi_tr/tr_train /data/kaldi_tr/log/train /data/kaldi_tr/mfcc/train
   steps/make_mfcc.sh --nj 4 --cmd "run.pl" /data/kaldi_tr/tr_dev /data/kaldi_tr/log/dev /data/kaldi_tr/mfcc/dev
   steps/make_mfcc.sh --nj 4 --cmd "run.pl" /data/kaldi_tr/tr_test /data/kaldi_tr/log/test /data/kaldi_tr/mfcc/test
   steps/compute_cmvn_stats.sh /data/kaldi_tr/tr_train /data/kaldi_tr/log/cmvn_train /data/kaldi_tr/mfcc/train
   steps/compute_cmvn_stats.sh /data/kaldi_tr/tr_dev /data/kaldi_tr/log/cmvn_dev /data/kaldi_tr/mfcc/dev
   steps/compute_cmvn_stats.sh /data/kaldi_tr/tr_test /data/kaldi_tr/log/cmvn_test /data/kaldi_tr/mfcc/test

6. python3 /data/proje/kaldi_lexicon_olustur.py

7. python3 /data/proje/kaldi_dict_hazirla.py
```