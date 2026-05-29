import os
import shutil

VERI_KLASORU = "database"
MFA_KLASORU = "mfa_input"
KELIMELER = ["baslat", "geri", "yukari"]

os.makedirs(MFA_KLASORU, exist_ok=True)

for kelime in KELIMELER:
    hedef_klasor = os.path.join(MFA_KLASORU, kelime)
    os.makedirs(hedef_klasor, exist_ok=True)
    
    kaynak = os.path.join(VERI_KLASORU, kelime)
    dosyalar = sorted([f for f in os.listdir(kaynak) if f.endswith(".wav")])
    
    for dosya in dosyalar:
        # wav kopyala
        shutil.copy(
            os.path.join(kaynak, dosya),
            os.path.join(hedef_klasor, dosya)
        )
        
        # transkript txt oluştur
        txt_adi = dosya.replace(".wav", ".txt")
        with open(os.path.join(hedef_klasor, txt_adi), "w") as f:
            f.write(kelime)
    
    print(f"{kelime}: {len(dosyalar)} wav + {len(dosyalar)} txt oluşturuldu")

print("\nmfa_input klasörü hazır!")
