import os
from phonemizer import phonemize

# text dosyasından tüm kelimeleri topla
TEXT_DOSYA = "/data/kaldi_tr/tr_train/text"
LEXICON_DIR = "/data/kaldi_tr/local/dict"

os.makedirs(LEXICON_DIR, exist_ok=True)

print("Kelimeler toplanıyor...")
kelimeler = set()
with open(TEXT_DOSYA, "r", encoding="utf-8") as f:
    for satir in f:
        parcalar = satir.strip().split()
        # İlk parça utterance_id, gerisí kelimeler
        for kelime in parcalar[1:]:
            kelimeler.add(kelime)

print(f"  {len(kelimeler)} farklı kelime bulundu")

# Her kelime için fonem üret
print("Fonemler üretiliyor...")
kelime_listesi = sorted(kelimeler)
fonemler = phonemize(
    kelime_listesi,
    language='tr',
    backend='espeak',
    with_stress=False,
    language_switch='remove-flags'
)

# lexicon.txt yaz
print("Lexicon yazılıyor...")
with open(os.path.join(LEXICON_DIR, "lexicon.txt"), "w", encoding="utf-8") as f:
    # Sessizlik için özel giriş
    f.write("<SIL> SIL\n")
    f.write("<UNK> SPN\n")
    
    for kelime, fonem in zip(kelime_listesi, fonemler):
        fonem = fonem.strip()
        if fonem:
            # Fonemleri boşlukla ayır
            fonem_listesi = " ".join(list(fonem.replace(" ", "")))
            # IPA → Kaldi ASCII fonem dönüşümü
            fonem_listesi = fonem_listesi.replace("ɫ", "l")
            fonem_listesi = fonem_listesi.replace("ʒ", "zh")
            fonem_listesi = fonem_listesi.replace("ɛ", "e")
            fonem_listesi = fonem_listesi.replace("ɯ", "ii")
            fonem_listesi = fonem_listesi.replace("ʃ", "sh")
            fonem_listesi = fonem_listesi.replace("ɾ", "r")
            fonem_listesi = fonem_listesi.replace("ɟ", "j")
            fonem_listesi = fonem_listesi.replace("t̪", "t")
            fonem_listesi = fonem_listesi.replace("d̪", "d")
            fonem_listesi = fonem_listesi.replace("n̪", "n")
            fonem_listesi = fonem_listesi.replace("s̪", "s")
            fonem_listesi = fonem_listesi.replace("ʎ", "ly")
            fonem_listesi = fonem_listesi.replace("æ", "a")
            fonem_listesi = fonem_listesi.replace("ø", "oe")
            fonem_listesi = fonem_listesi.replace("y", "ue")
            fonem_listesi = fonem_listesi.replace("œ", "oe")
            fonem_listesi = fonem_listesi.replace("ɔ", "o")
            fonem_listesi = fonem_listesi.replace("ɡ", "g")
            fonem_listesi = fonem_listesi.replace("ɪ", "i")
            fonem_listesi = fonem_listesi.replace("ʊ", "u")
            fonem_listesi = fonem_listesi.replace("ː", "")
            f.write(f"{kelime} {fonem_listesi}\n")

print(f"Lexicon hazır: {LEXICON_DIR}/lexicon.txt")

# İlk 10 satırı göster
print("\nİlk 10 giriş:")
with open(os.path.join(LEXICON_DIR, "lexicon.txt"), "r") as f:
    for i, satir in enumerate(f):
        if i >= 10:
            break
        print(f"  {satir.strip()}")
