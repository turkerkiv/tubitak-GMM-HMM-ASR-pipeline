import os
import csv
import random
import re

# ---------------------------------------------
# AYARLAR
# ---------------------------------------------
CV_DIR    = "/data/commonvoice"
KALDI_DIR = "/data/kaldi_tr"
CLIPS_DIR = os.path.join(CV_DIR, "clips")

# Rastgelelik sabit olsun
RANDOM_SEED = 42

os.makedirs(KALDI_DIR, exist_ok=True)


# ---------------------------------------------
# TSV OKU
# ---------------------------------------------
def tsv_oku(dosya):
    ornekler = []

    with open(dosya, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for satir in reader:
            mp3_yolu = os.path.join(CLIPS_DIR, satir["path"])

            # ses dosyası gerçekten var mı
            if os.path.exists(mp3_yolu):
                ornekler.append(satir)

    return ornekler


# ---------------------------------------------
# TRANSKRIPT TEMIZLE
# ---------------------------------------------
def temizle_metin(metin):
    metin = metin.strip().upper()

    # noktalama temizle
    metin = re.sub(r"[^\w\s]", "", metin)

    # fazla boşluk temizle
    metin = re.sub(r"\s+", " ", metin)

    return metin.strip()


# ---------------------------------------------
# KALDI DOSYALARI
# ---------------------------------------------
def kaldi_dosyalari_olustur(ornekler, cikti_klasoru):
    os.makedirs(cikti_klasoru, exist_ok=True)

    text_f  = open(os.path.join(cikti_klasoru, "text"), "w", encoding="utf-8")
    wav_f   = open(os.path.join(cikti_klasoru, "wav.scp"), "w", encoding="utf-8")
    utt2spk = open(os.path.join(cikti_klasoru, "utt2spk"), "w", encoding="utf-8")

    utt_listesi = []

    for satir in ornekler:

        # speaker id
        spk_id = satir["client_id"][:8]

        # utterance id
        utt_id = (
            spk_id + "-"
            + satir["path"]
                .replace(".mp3", "")
                .replace("/", "-")
        )

        # transcript
        metin = temizle_metin(satir["sentence"])

        # boş transcript varsa geç
        if len(metin) == 0:
            continue

        mp3_yolu = os.path.join(CLIPS_DIR, satir["path"])

        # Kaldi için sox pipe
        wav_komutu = (
            f"sox {mp3_yolu} "
            f"-t wav -r 16000 -b 16 -e signed - |"
        )

        text_f.write(f"{utt_id} {metin}\n")
        wav_f.write(f"{utt_id} {wav_komutu}\n")
        utt2spk.write(f"{utt_id} {spk_id}\n")

        utt_listesi.append((utt_id, spk_id))

    text_f.close()
    wav_f.close()
    utt2spk.close()

    # spk2utt
    spk2utt = {}

    for utt_id, spk_id in utt_listesi:
        if spk_id not in spk2utt:
            spk2utt[spk_id] = []

        spk2utt[spk_id].append(utt_id)

    with open(os.path.join(cikti_klasoru, "spk2utt"), "w") as f:
        for spk_id, uttler in sorted(spk2utt.items()):
            f.write(f"{spk_id} {' '.join(sorted(uttler))}\n")

    print(f"{cikti_klasoru} -> {len(utt_listesi)} örnek")


# ---------------------------------------------
# ANA AKIŞ
# ---------------------------------------------
print("validated.tsv okunuyor...")

validated_yolu = os.path.join(CV_DIR, "validated.tsv")

tum_veri = tsv_oku(validated_yolu)

print(f"Toplam örnek: {len(tum_veri)}")

# shuffle
random.seed(RANDOM_SEED)
random.shuffle(tum_veri)

# ---------------------------------------------
# SPLIT ORANLARI
# ---------------------------------------------
toplam = len(tum_veri)

train_oran = 0.90
dev_oran   = 0.05
test_oran  = 0.05

train_son = int(toplam * train_oran)
dev_son   = train_son + int(toplam * dev_oran)

train_data = tum_veri[:train_son]
dev_data   = tum_veri[train_son:dev_son]
test_data  = tum_veri[dev_son:]

print()
print(f"TRAIN: {len(train_data)}")
print(f"DEV  : {len(dev_data)}")
print(f"TEST : {len(test_data)}")

# ---------------------------------------------
# KALDI DOSYALARI YAZ
# ---------------------------------------------
kaldi_dosyalari_olustur(
    train_data,
    os.path.join(KALDI_DIR, "tr_train")
)

kaldi_dosyalari_olustur(
    dev_data,
    os.path.join(KALDI_DIR, "tr_dev")
)

kaldi_dosyalari_olustur(
    test_data,
    os.path.join(KALDI_DIR, "tr_test")
)

print()
print("Hazır!")
print(f"{KALDI_DIR}/tr_train")
print(f"{KALDI_DIR}/tr_dev")
print(f"{KALDI_DIR}/tr_test")