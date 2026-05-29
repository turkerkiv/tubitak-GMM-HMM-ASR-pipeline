#!/bin/bash

echo "=== Kaldi Türkçe ASR Başlatılıyor ==="

cd /opt/kaldi/egs/commonvoice/s5
source ./path.sh

# KenLM PATH
export PATH=/opt/kaldi/tools/kenlm/build/bin:$PATH

NJ=8

echo ""
echo "1. Veri hazırlanıyor..."
python3 /data/proje/kaldi_data_prep.py

echo ""
echo "2. Dosyalar sıralanıyor..."
for dir in /data/kaldi_tr/tr_train /data/kaldi_tr/tr_dev /data/kaldi_tr/tr_test; do
    sort $dir/utt2spk -o $dir/utt2spk
    sort $dir/text    -o $dir/text
    sort $dir/wav.scp -o $dir/wav.scp
done

echo ""
echo "3. MFCC çıkarılıyor..."
steps/make_mfcc.sh --nj $NJ --cmd "run.pl" /data/kaldi_tr/tr_train /data/kaldi_tr/log/train /data/kaldi_tr/mfcc/train
steps/make_mfcc.sh --nj $NJ --cmd "run.pl" /data/kaldi_tr/tr_dev   /data/kaldi_tr/log/dev   /data/kaldi_tr/mfcc/dev
steps/make_mfcc.sh --nj $NJ --cmd "run.pl" /data/kaldi_tr/tr_test  /data/kaldi_tr/log/test  /data/kaldi_tr/mfcc/test

echo ""
echo "4. CMVN hesaplanıyor..."
steps/compute_cmvn_stats.sh /data/kaldi_tr/tr_train /data/kaldi_tr/log/cmvn_train /data/kaldi_tr/mfcc/train
steps/compute_cmvn_stats.sh /data/kaldi_tr/tr_dev   /data/kaldi_tr/log/cmvn_dev   /data/kaldi_tr/mfcc/dev
steps/compute_cmvn_stats.sh /data/kaldi_tr/tr_test  /data/kaldi_tr/log/cmvn_test  /data/kaldi_tr/mfcc/test

echo ""
echo "5. Lexicon ve sözlük dosyaları oluşturuluyor..."
python3 /data/proje/kaldi_lexicon_olustur.py
python3 /data/proje/kaldi_dict_hazirla.py

echo ""
echo "6. Lang klasörü hazırlanıyor..."
utils/prepare_lang.sh \
    /data/kaldi_tr/local/dict \
    "<UNK>" \
    /data/kaldi_tr/local/lang \
    /data/kaldi_tr/lang

echo ""
echo "7. N-gram dil modeli eğitiliyor..."
awk '{$1=""; print $0}' /data/kaldi_tr/tr_train/text > /data/kaldi_tr/local/lm_train.txt

lmplz \
    -o 3 \
    --text /data/kaldi_tr/local/lm_train.txt \
    --arpa /data/kaldi_tr/local/lm.arpa

build_binary \
    /data/kaldi_tr/local/lm.arpa \
    /data/kaldi_tr/local/lm.bin

gzip -k -f /data/kaldi_tr/local/lm.arpa

utils/format_lm.sh \
    /data/kaldi_tr/lang \
    /data/kaldi_tr/local/lm.arpa.gz \
    /data/kaldi_tr/local/dict/lexicon.txt \
    /data/kaldi_tr/lang_test

echo ""
echo "8. Monophone GMM-HMM eğitiliyor..."
steps/train_mono.sh \
    --nj $NJ \
    --cmd "run.pl" \
    /data/kaldi_tr/tr_train \
    /data/kaldi_tr/lang \
    /data/kaldi_tr/exp/mono

echo ""
echo "9. Monophone decode..."
utils/mkgraph.sh \
    /data/kaldi_tr/lang_test \
    /data/kaldi_tr/exp/mono \
    /data/kaldi_tr/exp/mono/graph

steps/decode.sh \
    --nj $NJ \
    --cmd "run.pl" \
    /data/kaldi_tr/exp/mono/graph \
    /data/kaldi_tr/tr_test \
    /data/kaldi_tr/exp/mono/decode_test

echo "Monophone WER:"
cat /data/kaldi_tr/exp/mono/decode_test/scoring_kaldi/best_wer

echo ""
echo "10. Monophone hizalama..."
steps/align_si.sh \
    --nj $NJ \
    --cmd "run.pl" \
    /data/kaldi_tr/tr_train \
    /data/kaldi_tr/lang \
    /data/kaldi_tr/exp/mono \
    /data/kaldi_tr/exp/mono_ali

echo ""
echo "11. Triphone (tri1) eğitiliyor..."
steps/train_deltas.sh \
    --cmd "run.pl" \
    2000 10000 \
    /data/kaldi_tr/tr_train \
    /data/kaldi_tr/lang \
    /data/kaldi_tr/exp/mono_ali \
    /data/kaldi_tr/exp/tri1

echo ""
echo "12. Triphone (tri1) decode..."
utils/mkgraph.sh \
    /data/kaldi_tr/lang_test \
    /data/kaldi_tr/exp/tri1 \
    /data/kaldi_tr/exp/tri1/graph

steps/decode.sh \
    --nj $NJ \
    --cmd "run.pl" \
    /data/kaldi_tr/exp/tri1/graph \
    /data/kaldi_tr/tr_test \
    /data/kaldi_tr/exp/tri1/decode_test

echo "Triphone (tri1) WER:"
cat /data/kaldi_tr/exp/tri1/decode_test/scoring_kaldi/best_wer

echo ""
echo "13. Triphone (tri1) hizalama..."
steps/align_si.sh \
    --nj $NJ \
    --cmd "run.pl" \
    /data/kaldi_tr/tr_train \
    /data/kaldi_tr/lang \
    /data/kaldi_tr/exp/tri1 \
    /data/kaldi_tr/exp/tri1_ali

echo ""
echo "14. LDA+MLLT (tri2) eğitiliyor..."
steps/train_lda_mllt.sh \
    --cmd "run.pl" \
    2500 15000 \
    /data/kaldi_tr/tr_train \
    /data/kaldi_tr/lang \
    /data/kaldi_tr/exp/tri1_ali \
    /data/kaldi_tr/exp/tri2

echo ""
echo "15. LDA+MLLT (tri2) decode..."
utils/mkgraph.sh \
    /data/kaldi_tr/lang_test \
    /data/kaldi_tr/exp/tri2 \
    /data/kaldi_tr/exp/tri2/graph

steps/decode.sh \
    --nj $NJ \
    --cmd "run.pl" \
    /data/kaldi_tr/exp/tri2/graph \
    /data/kaldi_tr/tr_test \
    /data/kaldi_tr/exp/tri2/decode_test

echo "LDA+MLLT (tri2) WER:"
cat /data/kaldi_tr/exp/tri2/decode_test/scoring_kaldi/best_wer

echo ""
echo "16. LDA+MLLT (tri2) hizalama..."
steps/align_si.sh \
    --nj $NJ \
    --cmd "run.pl" \
    /data/kaldi_tr/tr_train \
    /data/kaldi_tr/lang \
    /data/kaldi_tr/exp/tri2 \
    /data/kaldi_tr/exp/tri2_ali

echo ""
echo "17. SAT (tri3) eğitiliyor (4000 durum, 40000 Gaussian)..."
steps/train_sat.sh \
    --cmd "run.pl" \
    4000 40000 \
    /data/kaldi_tr/tr_train \
    /data/kaldi_tr/lang \
    /data/kaldi_tr/exp/tri2_ali \
    /data/kaldi_tr/exp/tri3

echo ""
echo "18. SAT (tri3) decode..."
utils/mkgraph.sh \
    /data/kaldi_tr/lang_test \
    /data/kaldi_tr/exp/tri3 \
    /data/kaldi_tr/exp/tri3/graph

steps/decode_fmllr.sh \
    --nj $NJ \
    --cmd "run.pl" \
    /data/kaldi_tr/exp/tri3/graph \
    /data/kaldi_tr/tr_test \
    /data/kaldi_tr/exp/tri3/decode_test

echo "SAT (tri3) WER:"
cat /data/kaldi_tr/exp/tri3/decode_test/scoring_kaldi/best_wer

echo ""
echo "=== TÜM SONUÇLAR ==="
echo "Monophone WER:"
cat /data/kaldi_tr/exp/mono/decode_test/scoring_kaldi/best_wer
echo "Triphone (tri1) WER:"
cat /data/kaldi_tr/exp/tri1/decode_test/scoring_kaldi/best_wer
echo "LDA+MLLT (tri2) WER:"
cat /data/kaldi_tr/exp/tri2/decode_test/scoring_kaldi/best_wer
echo "SAT (tri3) WER:"
cat /data/kaldi_tr/exp/tri3/decode_test/scoring_kaldi/best_wer

echo ""
echo "=== Hazır! Bash'e düşülüyor... ==="
exec bash
