import os
import numpy as np
import librosa
import textgrid
from collections import defaultdict

KELIMELER = {
    "baslat": "başlat",
    "geri": "geri", 
    "yukari": "yukarı"
}

N_MFCC = 13

# ---------------------------------------------
# ADIM 1: TÜM TEXTGRID + SES DOSYALARINI OKU
# HER FONEM İÇİN MFCC KESİTİ ÇIKAR
# ---------------------------------------------
print("Fonem MFCC'leri çıkarılıyor...")

# fonem_veri["b"] = [mfcc1, mfcc2, ...]  şeklinde
fonem_veri = defaultdict(list)

for klasor_adi, kelime in KELIMELER.items():
    wav_klasoru = f"database/{klasor_adi}"
    tg_klasoru  = f"mfa_output/{klasor_adi}"
    
    wav_dosyalar = sorted([f for f in os.listdir(wav_klasoru) if f.endswith(".wav")])
    
    for wav_dosya in wav_dosyalar:
        wav_yolu = os.path.join(wav_klasoru, wav_dosya)
        tg_yolu  = os.path.join(tg_klasoru, wav_dosya.replace(".wav", ".TextGrid"))
        
        if not os.path.exists(tg_yolu):
            continue
        
        # Ses yükle
        y, sr = librosa.load(wav_yolu, sr=None)
        
        # TextGrid oku
        tg = textgrid.TextGrid.fromFile(tg_yolu)
        
        for tier in tg.tiers:
            if tier.name != "phones":
                continue
            
            for interval in tier:
                fonem = interval.mark.strip()
                if not fonem or fonem in ["spn", "sil", ""]:
                    continue
                
                # Zaman → örnek indeksine çevir
                baslangic = int(interval.minTime * sr)
                bitis     = int(interval.maxTime * sr)
                
                # O foneme ait ses kesiti
                fonem_ses = y[baslangic:bitis]
                
                # Çok kısaysa atla (gürültü olabilir)
                if len(fonem_ses) < 100:
                    continue
                
                # MFCC çıkar
                n_fft = min(512, len(fonem_ses))
                mfcc = librosa.feature.mfcc(y=fonem_ses, sr=sr, n_mfcc=N_MFCC, n_fft=n_fft)
                
                if mfcc.shape[1] < 2:
                    continue
                
                fonem_veri[fonem].append(mfcc.T)

# ---------------------------------------------
# ADIM 2: SONUÇLARI GÖSTER
# ---------------------------------------------
print("\n--- BULUNAN FONEMLER ---")
for fonem, mfcc_listesi in sorted(fonem_veri.items()):
    ortalama_uzunluk = np.mean([len(m) for m in mfcc_listesi])
    print(f"  '{fonem}': {len(mfcc_listesi)} örnek, "
          f"ortalama {ortalama_uzunluk:.1f} frame")

print(f"\nToplam farklı fonem: {len(fonem_veri)}")

# ---------------------------------------------
# ADIM 3: KAYDET (Toy9'da kullanacağız)
# ---------------------------------------------
np.save("fonem_veri.npy", dict(fonem_veri))
print("\nfonem_veri.npy kaydedildi, Toy9'da kullanacağız!")
