import os
import numpy as np
import librosa
from hmmlearn import hmm
from phonemizer import phonemize

# ---------------------------------------------
# AYARLAR
# ---------------------------------------------
VERI_KLASORU = "database"
KELIMELER = ["baslat", "geri", "yukari"]
N_MFCC = 13
N_ITER = 100

# ---------------------------------------------
# ADIM 1: KELİMELERİ FONEMLERE ÇEVİR
# ---------------------------------------------
print("Kelimeler fonemlere çevriliyor...")

kelime_fonem = {}
fonem_seti = set()

for kelime in KELIMELER:
    fonemler = phonemize(kelime, language='tr', backend='espeak')
    # boşlukları temizle, her karakter bir fonem
    fonem_listesi = list(fonemler.strip())
    kelime_fonem[kelime] = fonem_listesi
    fonem_seti.update(fonem_listesi)
    print(f"  {kelime} → {fonem_listesi}")

print(f"\nToplam farklı fonem sayısı: {len(fonem_seti)}")
print(f"Fonemler: {sorted(fonem_seti)}")

# ---------------------------------------------
# ADIM 2: SES DOSYALARINI YÜKLE VE MFCC ÇIKAR
# ---------------------------------------------
print("\nMFCC çıkarılıyor...")

def mfcc_cikar(dosya_yolu):
    y, sr = librosa.load(dosya_yolu, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return mfcc.T

veri = {}
for kelime in KELIMELER:
    klasor = os.path.join(VERI_KLASORU, kelime)
    dosyalar = sorted([f for f in os.listdir(klasor) if f.endswith(".wav")])
    
    mfcc_listesi = []
    for dosya in dosyalar:
        yol = os.path.join(klasor, dosya)
        mfcc_listesi.append(mfcc_cikar(yol))
    
    veri[kelime] = mfcc_listesi
    print(f"  {kelime}: {len(mfcc_listesi)} dosya")

# ---------------------------------------------
# ADIM 3: HER KELİME İÇİN FONEM SAYISI KADAR
#          GİZLİ DURUM İÇEREN HMM EĞİT
# ---------------------------------------------
print("\nHMM modelleri eğitiliyor...")

modeller = {}

for kelime in KELIMELER:
    # Fonem sayısı kadar gizli durum kullan
    # "baslat" = 6 fonem → 6 gizli durum
    n_durum = len(kelime_fonem[kelime])
    print(f"  {kelime}: {n_durum} fonem → {n_durum} gizli durum")
    
    model = hmm.GaussianHMM(
        n_components=n_durum,
        covariance_type="diag",
        n_iter=N_ITER
    )
    
    # Eğitim için 7 dosya kullan
    egitim = veri[kelime][:7]
    X = np.concatenate(egitim)
    lengths = [len(m) for m in egitim]
    
    model.fit(X, lengths)
    modeller[kelime] = model

# ---------------------------------------------
# ADIM 4: TEST ET
# ---------------------------------------------
print("\n--- TEST SONUÇLARI ---")

dogru = 0
toplam = 0

for gercek_kelime in KELIMELER:
    # Son 3 dosyayı test olarak kullan
    for test_mfcc in veri[gercek_kelime][7:]:
        skorlar = {}
        for kelime, model in modeller.items():
            skorlar[kelime] = model.score(test_mfcc)
        
        tahmin = max(skorlar, key=skorlar.get)
        sonuc = "✅" if tahmin == gercek_kelime else "❌"
        print(f"  Gerçek: {gercek_kelime:10} → Tahmin: {tahmin:10} {sonuc}")
        
        if tahmin == gercek_kelime:
            dogru += 1
        toplam += 1

print(f"\nDoğruluk: {dogru}/{toplam} = %{dogru/toplam*100:.1f}")

print("\n--- FARK NE? ---")
print("Toy4: Her kelime için sabit 3 gizli durum")
print("Toy6: Her kelime için fonem sayısı kadar gizli durum")
for kelime in KELIMELER:
    print(f"  {kelime}: {len(kelime_fonem[kelime])} durum {kelime_fonem[kelime]}")
