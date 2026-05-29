#!/bin/bash

echo "=== Kaldi Türkçe ASR Başlatılıyor ==="

# Path ayarla
cd /opt/kaldi/egs/commonvoice/s5
source ./path.sh

# Veri hazırla
echo "1. Veri hazırlanıyor..."
python3 /data/proje/kaldi_data_prep.py

# Sırala
echo "2. Dosyalar sıralanıyor..."
for dir in /data/kaldi_tr/tr_train /data/kaldi_tr/tr_dev /data/kaldi_tr/tr_test; do
  sort $dir/utt2spk -o $dir/utt2spk
  sort $dir/text -o $dir/text
  sort $dir/wav.scp -o $dir/wav.scp
done

# MFCC
echo "3. MFCC çıkarılıyor..."
steps/make_mfcc.sh --nj 4 --cmd "run.pl" /data/kaldi_tr/tr_train /data/kaldi_tr/log/train /data/kaldi_tr/mfcc/train
steps/make_mfcc.sh --nj 4 --cmd "run.pl" /data/kaldi_tr/tr_dev /data/kaldi_tr/log/dev /data/kaldi_tr/mfcc/dev
steps/make_mfcc.sh --nj 4 --cmd "run.pl" /data/kaldi_tr/tr_test /data/kaldi_tr/log/test /data/kaldi_tr/mfcc/test

# CMVN
echo "4. CMVN hesaplanıyor..."
steps/compute_cmvn_stats.sh /data/kaldi_tr/tr_train /data/kaldi_tr/log/cmvn_train /data/kaldi_tr/mfcc/train
steps/compute_cmvn_stats.sh /data/kaldi_tr/tr_dev /data/kaldi_tr/log/cmvn_dev /data/kaldi_tr/mfcc/dev
steps/compute_cmvn_stats.sh /data/kaldi_tr/tr_test /data/kaldi_tr/log/cmvn_test /data/kaldi_tr/mfcc/test

# Lexicon
echo "5. Lexicon oluşturuluyor..."
python3 /data/proje/kaldi_lexicon_olustur.py
python3 /data/proje/kaldi_dict_hazirla.py

echo "=== Hazır! ==="
exec bash