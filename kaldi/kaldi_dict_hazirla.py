import os

LEXICON_DIR = "/data/kaldi_tr/local/dict"

# Lexicon'dan tüm fonemleri topla
print("Fonemler toplanıyor...")
fonemler = set()
with open(os.path.join(LEXICON_DIR, "lexicon.txt"), "r") as f:
    for satir in f:
        parcalar = satir.strip().split()
        if len(parcalar) > 1:
            for fonem in parcalar[1:]:
                fonemler.add(fonem)

# Sessizlik fonemleri
sessizlik = {"SIL", "SPN"}
sessizlik_olmayan = fonemler - sessizlik

print(f"  Sessizlik fonemleri    : {sessizlik}")
print(f"  Sessizlik olmayan fonem: {len(sessizlik_olmayan)}")

# silence_phones.txt
with open(os.path.join(LEXICON_DIR, "silence_phones.txt"), "w") as f:
    for fonem in sorted(sessizlik):
        f.write(f"{fonem}\n")

# nonsilence_phones.txt
with open(os.path.join(LEXICON_DIR, "nonsilence_phones.txt"), "w") as f:
    for fonem in sorted(sessizlik_olmayan):
        f.write(f"{fonem}\n")

# optional_silence.txt
with open(os.path.join(LEXICON_DIR, "optional_silence.txt"), "w") as f:
    f.write("SIL\n")

print("Dosyalar hazır!")
print("\nNonsilence fonemler:")
for f in sorted(sessizlik_olmayan):
    print(f"  {f}")
