import os
import numpy as np
import librosa
import textgrid
from hmmlearn import hmm
from collections import defaultdict

KELIMELER = {
    "baslat": "başlat",
    "geri": "geri",
    "yukari": "yukarı"
}

N_MFCC = 13
N_ITER = 100

# ---------------------------------------------
# ADIM 1: HER KELİME İÇİN MFCC + FONEM SINIRLARINI OKU
# ---------------------------------------------
print("Veri hazırlanıyor...")

# kelime_veri["başlat"] = [
#     {
#       "mfcc": (zaman, 13) matris,
#       "fonemler": ["b","a","ʃ","ɫ","a","t̪"]
#     },
#     ...
# ]
kelime_veri = defaultdict(list)

for klasor_adi, kelime in KELIMELER.items():
    wav_klasoru = f"database/{klasor_adi}"
    tg_klasoru  = f"mfa_output/{klasor_adi}"
    
    wav_dosyalar = sorted([f for f in os.listdir(wav_klasoru) if f.endswith(".wav")])
    
    for wav_dosya in wav_dosyalar:
        wav_yolu = os.path.join(wav_klasoru, wav_dosya)
        tg_yolu  = os.path.join(tg_klasoru, wav_dosya.replace(".wav", ".TextGrid"))
        
        if not os.path.exists(tg_yolu):
            continue
        
        # Tüm kelimeyi yükle, MFCC çıkar
        y, sr = librosa.load(wav_yolu, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T
        
        # TextGrid'den fonem listesini al
        tg = textgrid.TextGrid.fromFile(tg_yolu)
        fonemler = []
        for tier in tg.tiers:
            if tier.name != "phones":
                continue
            for interval in tier:
                fonem = interval.mark.strip()
                if fonem and fonem not in ["spn", "sil"]:
                    fonemler.append(fonem)
        
        if not fonemler:
            continue
        
        kelime_veri[kelime].append({
            "mfcc": mfcc,
            "fonemler": fonemler
        })
    
    print(f"  {kelime}: {len(kelime_veri[kelime])} dosya")

# ---------------------------------------------
# ADIM 2: HER KELİME İÇİN HMM EĞİT
# Fonem sayısı kadar gizli durum kullan
# ---------------------------------------------
print("\nHMM modelleri eğitiliyor...")

modeller = {}

for kelime, ornekler in kelime_veri.items():
    # Fonem sayısını ilk örnekten al
    n_durum = len(ornekler[0]["fonemler"])
    fonemler = ornekler[0]["fonemler"]
    
    print(f"\n  {kelime}:")
    print(f"    Fonemler : {fonemler}")
    print(f"    Durum    : {n_durum}")
    print(f"    Örnek    : {len(ornekler)}")
    
    # Eğitim için ilk 7 örnek
    egitim = ornekler[:7]
    
    # Tüm MFCC'leri birleştir
    X = np.concatenate([o["mfcc"] for o in egitim])
    lengths = [len(o["mfcc"]) for o in egitim]
    
    model = hmm.GaussianHMM(
        n_components=n_durum,
        covariance_type="diag",
        n_iter=N_ITER
    )
    model.fit(X, lengths)
    modeller[kelime] = model

# ---------------------------------------------
# ADIM 3: TEST ET
# ---------------------------------------------
print("\n--- TEST SONUÇLARI ---")

dogru = 0
toplam = 0

for gercek_kelime, ornekler in kelime_veri.items():
    # Son 3 örnek test
    for ornek in ornekler[7:]:
        skorlar = {}
        for kelime, model in modeller.items():
            skorlar[kelime] = model.score(ornek["mfcc"])
        
        tahmin = max(skorlar, key=skorlar.get)
        sonuc = "✅" if tahmin == gercek_kelime else "❌"
        print(f"  Gerçek: {gercek_kelime:10} → Tahmin: {tahmin:10} {sonuc}")
        
        if tahmin == gercek_kelime:
            dogru += 1
        toplam += 1

print(f"\nDoğruluk: {dogru}/{toplam} = %{dogru/toplam*100:.1f}")

print("\n--- ÖNCEKİLERLE KARŞILAŞTIRMA ---")
print("Toy4: Kelime bazlı, sabit 3 durum")
print("Toy6: Kelime bazlı, fonem sayısı kadar durum")  
print("Toy9: Kelime bazlı, fonem sayısı kadar durum + gerçek fonem bilgisi")
