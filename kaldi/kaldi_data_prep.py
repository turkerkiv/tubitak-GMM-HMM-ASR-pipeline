import os
import csv

# ---------------------------------------------
# AYARLAR
# ---------------------------------------------
CV_DIR    = "/data/commonvoice"
KALDI_DIR = "/data/kaldi_tr"
CLIPS_DIR = os.path.join(CV_DIR, "clips")

# Kaç örnek kullanalım (başlangıç için küçük tutalım)
MAX_TRAIN = 40000
MAX_DEV   = 3000
MAX_TEST  = 3000

os.makedirs(KALDI_DIR, exist_ok=True)


def tsv_oku(dosya, max_satir=None):
    ornekler = []
    with open(dosya, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, satir in enumerate(reader):
            if max_satir and i >= max_satir:
                break
            # Ses dosyası var mı kontrol et
            mp3_yolu = os.path.join(CLIPS_DIR, satir["path"])
            if os.path.exists(mp3_yolu):
                ornekler.append(satir)
    return ornekler


def kaldi_dosyalari_olustur(ornekler, cikti_klasoru):
    os.makedirs(cikti_klasoru, exist_ok=True)

    text_f  = open(os.path.join(cikti_klasoru, "text"),    "w", encoding="utf-8")
    wav_f   = open(os.path.join(cikti_klasoru, "wav.scp"), "w", encoding="utf-8")
    utt2spk = open(os.path.join(cikti_klasoru, "utt2spk"), "w", encoding="utf-8")

    utt_listesi = []

    for satir in ornekler:
        # speaker id (client_id nin ilk 8 karakteri)
        spk_id = satir["client_id"][:8]

        # utterance id olustur
        utt_id = spk_id + "-" + satir["path"].replace(".mp3", "").replace("/", "-")

        # transkript temizle
        metin = satir["sentence"].strip().upper()
        # Noktalama temizle
        import re
        metin = re.sub(r'[^\w\s]', '', metin)  # harf ve boşluk dışını sil
        metin = metin.strip()

        mp3_yolu = os.path.join(CLIPS_DIR, satir["path"])

        # Kaldi mp3 okuyamaz, sox ile wava cevir
        wav_komutu = f"sox {mp3_yolu} -t wav -r 16000 -b 16 -e signed - |"

        text_f.write(f"{utt_id} {metin}\n")
        wav_f.write(f"{utt_id} {wav_komutu}\n")
        utt2spk.write(f"{utt_id} {spk_id}\n")
        utt_listesi.append((utt_id, spk_id))

    text_f.close()
    wav_f.close()
    utt2spk.close()

    # spk2utt olustur
    spk2utt = {}
    for utt_id, spk_id in utt_listesi:
        if spk_id not in spk2utt:
            spk2utt[spk_id] = []
        spk2utt[spk_id].append(utt_id)

    with open(os.path.join(cikti_klasoru, "spk2utt"), "w") as f:
        for spk_id, uttler in sorted(spk2utt.items()):
            f.write(f"{spk_id} {' '.join(sorted(uttler))}\n")

    print(f"  {len(ornekler)} ornek islendi -> {cikti_klasoru}")


# ---------------------------------------------
# TRAIN / DEV / TEST HAZIRLA
# ---------------------------------------------
print("Common Voice Turkce veri hazirlaniyor...")
print(f"  Train: {MAX_TRAIN} ornek")
print(f"  Dev  : {MAX_DEV} ornek")
print(f"  Test : {MAX_TEST} ornek")

bolumler = {
    "train": ("train.tsv", MAX_TRAIN, "tr_train"),
    "dev":   ("dev.tsv",   MAX_DEV,   "tr_dev"),
    "test":  ("test.tsv",  MAX_TEST,  "tr_test"),
}

for bolum, (tsv_dosya, max_s, cikti_adi) in bolumler.items():
    tsv_yolu = os.path.join(CV_DIR, tsv_dosya)
    if not os.path.exists(tsv_yolu):
        print(f"  {tsv_dosya} bulunamadi, atliyorum")
        continue

    ornekler = tsv_oku(tsv_yolu, max_satir=max_s)
    cikti    = os.path.join(KALDI_DIR, cikti_adi)
    kaldi_dosyalari_olustur(ornekler, cikti)

print("\nTamam! Kaldi veri dosyalari hazir:")
print(f"  {KALDI_DIR}/tr_train/")
print(f"  {KALDI_DIR}/tr_dev/")
print(f"  {KALDI_DIR}/tr_test/")
print("\nHer klasorde: text, wav.scp, utt2spk, spk2utt")
