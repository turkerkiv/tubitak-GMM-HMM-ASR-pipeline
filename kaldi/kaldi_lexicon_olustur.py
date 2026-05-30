import os
import csv
import re
from phonemizer import phonemize

SENTENCES_TSV = "/data/commonvoice/validated_sentences.tsv"
TRAIN_TEXT    = "/data/kaldi_tr/tr_train/text"
LEXICON_DIR   = "/data/kaldi_tr/local/dict"

os.makedirs(LEXICON_DIR, exist_ok=True)

# ---------------------------------------------
# ADIM 1: validated_sentences.tsv den kelimeleri topla
# ---------------------------------------------
print("=== ADIM 1: Kelimeler toplanıyor ===")

kelimeler = set()

# validated_sentences.tsv - sekmeyle ayrılmış, sentence 2. sütun
with open(SENTENCES_TSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    satir_sayisi = 0
    for satir in reader:
        cumle = satir["sentence"].strip().upper()
        cumle = re.sub(r'[^\w\s]', '', cumle)
        for kelime in cumle.split():
            if kelime:
                kelimeler.add(kelime)
        satir_sayisi += 1

print(f"  {satir_sayisi} cümle okundu")
print(f"  {len(kelimeler)} farklı kelime bulundu (validated_sentences)")

# Akustik eğitim metnindeki kelimeleri de ekle
# (bunlar lexiconda kesinlikle olmalı)
with open(TRAIN_TEXT, "r", encoding="utf-8") as f:
    for satir in f:
        parcalar = satir.strip().split()
        for kelime in parcalar[1:]:  # ilk parça utterance_id
            if kelime:
                kelimeler.add(kelime)

print(f"  Akustik veri eklendi: toplam {len(kelimeler)} kelime")

# ---------------------------------------------
# CHECKUP 1: Kelime örnekleri
# ---------------------------------------------
print("\n=== CHECKUP 1: Örnek kelimeler ===")
ornek = sorted(list(kelimeler))[:10]
for k in ornek:
    print(f"  {k}")

# ---------------------------------------------
# ADIM 2: Fonem üret
# ---------------------------------------------
print("\n=== ADIM 2: Fonemler üretiliyor ===")
print("  (Bu birkaç dakika sürebilir...)")

kelime_listesi = sorted(kelimeler)
fonemler = phonemize(
    kelime_listesi,
    language='tr',
    backend='espeak',
    with_stress=False,
    language_switch='remove-flags'
)

print(f"  {len(fonemler)} kelime için fonem üretildi")

# ---------------------------------------------
# ADIM 3: IPA → ASCII dönüşümü ve lexicon yaz
# ---------------------------------------------
print("\n=== ADIM 3: Lexicon yazılıyor ===")

IPA_MAP = {
    "ɫ": "l",  "ʒ": "zh", "ɛ": "e",  "ɯ": "ii",
    "ʃ": "sh", "ɾ": "r",  "ɟ": "j",  "t̪": "t",
    "d̪": "d",  "n̪": "n",  "s̪": "s",  "ʎ": "ly",
    "æ": "a",  "ø": "oe", "y": "ue", "œ": "oe",
    "ɔ": "o",  "ɡ": "g",  "ɪ": "i",  "ʊ": "u",
    "ː": "", "ɹ": "r", "ɣ": "g",
}

yazilan = 0
atlanan = 0
ipa_kalan = set()

with open(os.path.join(LEXICON_DIR, "lexicon.txt"), "w", encoding="utf-8") as f:
    f.write("<SIL> SIL\n")
    f.write("<UNK> SPN\n")

    for kelime, fonem in zip(kelime_listesi, fonemler):
        fonem = fonem.strip()
        if not fonem:
            atlanan += 1
            continue

        fonem_str = " ".join(list(fonem.replace(" ", "")))

        for ipa, ascii in IPA_MAP.items():
            fonem_str = fonem_str.replace(ipa, ascii)

        # Kalan IPA karakterleri tespit et
        for karakter in fonem_str.split():
            if not karakter.isascii():
                ipa_kalan.add(karakter)

        f.write(f"{kelime} {fonem_str}\n")
        yazilan += 1

print(f"  {yazilan} kelime yazıldı")
print(f"  {atlanan} kelime atlandı (boş fonem)")

# ---------------------------------------------
# CHECKUP 2: IPA karakter kaldı mı?
# ---------------------------------------------
print("\n=== CHECKUP 2: IPA karakter kontrolü ===")
if ipa_kalan:
    print(f"  UYARI: Hala IPA karakter var: {ipa_kalan}")
    print("  IPA_MAP e eklenmeli!")
else:
    print("  Tüm fonemler ASCII, IPA kalmadı")

# ---------------------------------------------
# CHECKUP 3: Lexicon örnek satırlar
# ---------------------------------------------
print("\n=== CHECKUP 3: Lexicon ilk 10 satır ===")
with open(os.path.join(LEXICON_DIR, "lexicon.txt"), "r") as f:
    for i, satir in enumerate(f):
        if i >= 12:
            break
        print(f"  {satir.strip()}")

# ---------------------------------------------
# CHECKUP 4: Akustik verideki kelimeler lexiconda var mı?
# ---------------------------------------------
print("\n=== CHECKUP 4: Akustik kelimeler lexiconda var mı? ===")

lexicon_kelimeler = set()
with open(os.path.join(LEXICON_DIR, "lexicon.txt"), "r") as f:
    for satir in f:
        parcalar = satir.strip().split()
        if parcalar:
            lexicon_kelimeler.add(parcalar[0])

eksik = []
with open(TRAIN_TEXT, "r", encoding="utf-8") as f:
    for satir in f:
        parcalar = satir.strip().split()
        for kelime in parcalar[1:]:
            if kelime and kelime not in lexicon_kelimeler:
                eksik.append(kelime)

if eksik:
    print(f"  UYARI: {len(eksik)} kelime lexiconda eksik!")
    print(f"  Örnek: {eksik[:5]}")
else:
    print(f"  Akustik verideki tüm kelimeler lexiconda mevcut")

print(f"\nLexicon hazır: {LEXICON_DIR}/lexicon.txt")
print(f"Toplam {yazilan + 2} satır (2 sessizlik + {yazilan} kelime)")