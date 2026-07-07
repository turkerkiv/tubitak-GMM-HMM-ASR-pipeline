#!/bin/bash
set -e

# ============================================================
# Kaldi ile bir KLASÖRDEKİ TÜM mp3'leri tanıma + WER + Sistem Metrikleri
# Kullanım: ./toplu_tanima.sh <mp3_klasoru> <tsv_dosyasi>
#   tsv_dosyasi: CommonVoice formatında, TAB-separated, başlık satırlı,
#                "path" ve "sentence" sütunlarını içermeli (test.tsv gibi)
# NOT: Container İÇİNDE çalıştırılmalıdır.
# ============================================================

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Kullanım: $0 <mp3_klasoru> <tsv_dosyasi>"
    exit 1
fi

MP3_DIR="$1"
TSV_FILE="$2"
MODEL_DIR=/data/proje/models/tdnn
WORK_DIR=/data/proje/toplu_tanima

T_BASLANGIC=$(date +%s)

cd /opt/kaldi/egs/commonvoice/s5
source ./path.sh
export PATH=/opt/kaldi/tools/kenlm/build/bin:$PATH

rm -rf $WORK_DIR
mkdir -p $WORK_DIR/wav

TESTDIR=$WORK_DIR/test
mkdir -p $TESTDIR
> $TESTDIR/wav.scp
> $TESTDIR/utt2spk
> $TESTDIR/text

echo ""
echo "1. tsv okunuyor (kaldi_data_prep.py ile AYNI normalizasyon uygulanıyor)..."

python3 <<PYEOF
import os, csv, re

MP3_DIR  = "$MP3_DIR"
TSV_FILE = "$TSV_FILE"
WORK_DIR = "$WORK_DIR"
TESTDIR  = "$TESTDIR"

wav_scp  = open(os.path.join(TESTDIR, "wav.scp"),  "w", encoding="utf-8")
utt2spk  = open(os.path.join(TESTDIR, "utt2spk"),  "w", encoding="utf-8")
text_f   = open(os.path.join(TESTDIR, "text"),     "w", encoding="utf-8")
mp3_list = open(os.path.join(WORK_DIR, "mp3_wav_list.txt"), "w", encoding="utf-8")

count = 0
with open(TSV_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for satir in reader:
        mp3_yolu = os.path.join(MP3_DIR, satir["path"])
        if not os.path.exists(mp3_yolu):
            continue

        uttid = os.path.basename(satir["path"]).replace(".mp3", "")
        wavpath = os.path.join(WORK_DIR, "wav", uttid + ".wav")

        # === kaldi_data_prep.py İLE BİREBİR AYNI NORMALİZASYON ===
        metin = satir["sentence"].strip().upper()
        metin = re.sub(r'[^\w\s]', '', metin)
        metin = metin.strip()

        if not metin:
            continue

        wav_scp.write(f"{uttid} {wavpath}\n")
        utt2spk.write(f"{uttid} spk_{uttid}\n")
        text_f.write(f"{uttid} {metin}\n")
        mp3_list.write(f"{mp3_yolu} {wavpath}\n")
        count += 1

wav_scp.close(); utt2spk.close(); text_f.close(); mp3_list.close()
print(f"  {count} dosya eşleşti ve işlendi.")
PYEOF

# mp3 -> wav dönüşümünü PARALEL yap
echo "mp3 -> wav dönüştürme (paralel, 16 işlem)..."
T_WAV_BASLA=$(date +%s)
xargs -P 16 -L 1 -a $WORK_DIR/mp3_wav_list.txt bash -c 'sox "$1" -r 16000 -c 1 "$2"' _
T_WAV_BITIS=$(date +%s)

sort $TESTDIR/wav.scp -o $TESTDIR/wav.scp
sort $TESTDIR/utt2spk -o $TESTDIR/utt2spk
sort $TESTDIR/text -o $TESTDIR/text
utils/utt2spk_to_spk2utt.pl $TESTDIR/utt2spk > $TESTDIR/spk2utt

N_UTT=$(wc -l < $TESTDIR/wav.scp)
echo "Toplam $N_UTT dosya bulundu ve işlendi."

if [ "$N_UTT" -eq 0 ]; then
    echo "HATA: Hiç eşleşen mp3 bulunamadı. Klasör yolunu ve tsv'deki 'path' sütununu kontrol edin."
    exit 1
fi

echo ""
echo "Toplam ses süresi hesaplanıyor..."
TOPLAM_SANIYE=$(cut -d' ' -f2 $WORK_DIR/mp3_wav_list.txt | xargs -P 16 -n1 soxi -D 2>/dev/null | awk '{sum+=$1} END {printf "%.2f", sum}')
TOPLAM_DAKIKA=$(awk -v s="$TOPLAM_SANIYE" 'BEGIN {printf "%.2f", s/60}')
echo "Toplam ses süresi: ${TOPLAM_SANIYE} saniye (${TOPLAM_DAKIKA} dakika)"

# nj, dosya sayısını geçmesin (16 çekirdeğe göre ayarlandı)
NJ=16
if [ "$N_UTT" -lt "$NJ" ]; then
    NJ=$N_UTT
fi

echo ""
echo "2. MFCC çıkarılıyor... (nj=$NJ)"
T_MFCC_BASLA=$(date +%s)
steps/make_mfcc.sh --nj $NJ --cmd "run.pl" \
    $TESTDIR $WORK_DIR/log $WORK_DIR/mfcc

echo ""
echo "3. CMVN hesaplanıyor..."
steps/compute_cmvn_stats.sh \
    $TESTDIR $WORK_DIR/log $WORK_DIR/mfcc
T_MFCC_BITIS=$(date +%s)

echo ""
echo "4. TDNN modeliyle decode ediliyor ve RAM/CPU kullanımı ölçülüyor..."
DECODE_DIR=$MODEL_DIR/decode_toplu
rm -rf $DECODE_DIR
T_DECODE_BASLA=$(date +%s)

# Peak RAM'i ölçmek için /usr/bin/time kullanıyoruz
if command -v /usr/bin/time >/dev/null 2>&1; then
    /usr/bin/time -o $WORK_DIR/ram_log.txt -v steps/nnet3/decode.sh --nj $NJ --cmd "run.pl" \
        $MODEL_DIR/graph \
        $TESTDIR \
        $DECODE_DIR
    
    PEAK_RAM_KB=$(grep "Maximum resident set size" $WORK_DIR/ram_log.txt | awk '{print $NF}')
    PEAK_RAM_MB=$(awk -v kb="$PEAK_RAM_KB" 'BEGIN {printf "%.2f", kb/1024}')
else
    steps/nnet3/decode.sh --nj $NJ --cmd "run.pl" \
        $MODEL_DIR/graph \
        $TESTDIR \
        $DECODE_DIR
    PEAK_RAM_MB="Ölçülemedi (/usr/bin/time komutu bulunamadı)"
fi

T_DECODE_BITIS=$(date +%s)
T_BITIS=$(date +%s)

# --- Sistem Metrikleri ve Süre Hesaplamaları ---
SURE_WAV=$((T_WAV_BITIS - T_WAV_BASLA))
SURE_MFCC=$((T_MFCC_BITIS - T_MFCC_BASLA))
SURE_DECODE=$((T_DECODE_BITIS - T_DECODE_BASLA))
SURE_TOPLAM=$((T_BITIS - T_BASLANGIC))

# RTF (Real-Time Factor)
RTF=$(awk -v d="$SURE_DECODE" -v s="$TOPLAM_SANIYE" 'BEGIN { if (s>0) printf "%.4f", d/s; else print "N/A" }')

# Throughput (Saniyede işlenen ses dosyası)
THROUGHPUT=$(awk -v n="$N_UTT" -v d="$SURE_DECODE" 'BEGIN { if (d>0) printf "%.2f", n/d; else print "N/A" }')

# Model Boyutları
MODEL_TOPLAM_BOYUT=$(du -sh $MODEL_DIR | cut -f1)
if [ -f "$MODEL_DIR/final.mdl" ]; then
    AKUSTIK_MODEL_BOYUT=$(du -sh $MODEL_DIR/final.mdl | cut -f1)
else
    AKUSTIK_MODEL_BOYUT="Bulunamadı"
fi

# --- En iyi WER dosyasını bul ---
BEST_WER_LINE=$(cat $DECODE_DIR/scoring_kaldi/best_wer)
BEST_WER_DOSYA=$(echo "$BEST_WER_LINE" | awk '{print $NF}')

if [ -f "$BEST_WER_DOSYA" ]; then
    TAM_SONUC=$(cat "$BEST_WER_DOSYA")
else
    TAM_SONUC="$BEST_WER_LINE"
fi

# --- Rapor ---
RAPOR=$WORK_DIR/rapor.txt

{
echo "================================================================"
echo "           KALDI TDNN MODELİ - PERFORMANS VE SİSTEM RAPORU"
echo "================================================================"
echo "Tarih                  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Test seti              : $TSV_FILE"
echo "Kullanılan dosya sayısı: $N_UTT"
echo "Paralel iş sayısı (nj) : $NJ"
echo ""
echo "--- DOĞRULUK (ACCURACY) ---"
echo "$TAM_SONUC"
echo ""
echo "--- KAYNAK TÜKETİMİ VE MODEL AĞIRLIĞI ---"
echo "Toplam Model Klasörü   : $MODEL_TOPLAM_BOYUT"
echo "Akustik Model (.mdl)   : $AKUSTIK_MODEL_BOYUT"
echo "Zirve (Peak) RAM Tük.  : $PEAK_RAM_MB MB"
echo ""
echo "--- SÜRE VE HIZ METRİKLERİ ---"
echo "Toplam ses süresi      : ${TOPLAM_SANIYE} sn (${TOPLAM_DAKIKA} dk)"
echo "mp3->wav çeviri süresi : ${SURE_WAV} sn"
echo "MFCC+CMVN süresi       : ${SURE_MFCC} sn"
echo "Decode süresi          : ${SURE_DECODE} sn"
echo "Toplam işlem süresi    : ${SURE_TOPLAM} sn"
echo ""
echo "--- PERFORMANS İNDEKSLERİ ---"
echo "RTF (Real-Time Factor) : ${RTF} (1'in altı = gerçek zamandan hızlı)"
echo "Throughput             : ${THROUGHPUT} dosya/saniye"
echo "================================================================"
} | tee $RAPOR

echo ""
echo "Rapor kaydedildi: $RAPOR"
echo "Detaylı sonuçlar için: $DECODE_DIR/scoring_kaldi/"
