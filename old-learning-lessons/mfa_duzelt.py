import os

DUZELTME = {
    "baslat": "başlat",
    "geri": "geri",
    "yukari": "yukarı"
}

MFA_KLASORU = "mfa_input"

for eski, yeni in DUZELTME.items():
    klasor = os.path.join(MFA_KLASORU, eski)
    if not os.path.exists(klasor):
        continue
    
    txt_dosyalar = [f for f in os.listdir(klasor) if f.endswith(".txt")]
    for txt in txt_dosyalar:
        yol = os.path.join(klasor, txt)
        with open(yol, "w", encoding="utf-8") as f:
            f.write(yeni)
    
    print(f"{eski} → {yeni} : {len(txt_dosyalar)} dosya güncellendi")

print("\nTamam!")
