import os
import numpy as np
import librosa
from hmmlearn import hmm
from sklearn.model_selection import train_test_split

# ---------------------------------------------
# AYARLAR
# ---------------------------------------------
VERI_KLASORU = "database"
KELIMELER = ["baslat", "geri", "yukari"]
N_MFCC = 13          # kaç MFCC katsayısı çıkaralım
N_COMPONENTS = 3     # HMM'deki gizli durum sayısı
N_ITER = 100         # eğitim tekrar sayısı

# ---------------------------------------------
# ADIM 1: SES DOSYALARINI YÜKLE VE MFCC ÇIKAR
# ---------------------------------------------
def mfcc_cikar(dosya_yolu):
    """Bir wav dosyasından MFCC çıkarır"""
    y, sr = librosa.load(dosya_yolu, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    # (13, zaman) → (zaman, 13) çeviriyoruz, HMM böyle istiyor
    return mfcc.T

print("Ses dosyaları yükleniyor ve MFCC çıkarılıyor...")

veri = {}   # her kelime için MFCC listesi

for kelime in KELIMELER:
    klasor = os.path.join(VERI_KLASORU, kelime)
    dosyalar = [f for f in os.listdir(klasor) if f.endswith(".wav")]
    
    mfcc_listesi = []
    for dosya in dosyalar:
        yol = os.path.join(klasor, dosya)
        mfcc = mfcc_cikar(yol)
        mfcc_listesi.append(mfcc)
    
    veri[kelime] = mfcc_listesi
    print(f"  {kelime}: {len(mfcc_listesi)} dosya yüklendi")

# ---------------------------------------------
# ADIM 2: EĞİTİM / TEST BÖL
# ---------------------------------------------
print("\nEğitim/test bölünüyor (7 eğitim, 3 test)...")

egitim_veri = {}
test_veri = {}

for kelime in KELIMELER:
    egitim, test = train_test_split(veri[kelime], test_size=3, random_state=42)
    egitim_veri[kelime] = egitim
    test_veri[kelime] = test
    print(f"  {kelime}: {len(egitim)} eğitim, {len(test)} test")

# ---------------------------------------------
# ADIM 3: HER KELİME İÇİN BİR HMM MODELI EĞİT
# ---------------------------------------------
print("\nHMM modelleri eğitiliyor...")

modeller = {}

for kelime in KELIMELER:
    model = hmm.GaussianHMM(
        n_components=N_COMPONENTS,
        covariance_type="diag",
        n_iter=N_ITER
    )
    
    # Tüm eğitim verilerini birleştir
    # HMM birleşik veri + uzunluk listesi istiyor
    X = np.concatenate(egitim_veri[kelime])
    lengths = [len(m) for m in egitim_veri[kelime]]
    
    model.fit(X, lengths)
    modeller[kelime] = model
    print(f"  {kelime} modeli eğitildi")

# ---------------------------------------------
# ADIM 4: TEST ET
# ---------------------------------------------
print("\n--- TEST SONUÇLARI ---")

dogru = 0
toplam = 0

for gercek_kelime in KELIMELER:
    for test_mfcc in test_veri[gercek_kelime]:
        # Her modelin bu sese verdiği skoru hesapla
        skorlar = {}
        for kelime, model in modeller.items():
            skorlar[kelime] = model.score(test_mfcc)
        
        # En yüksek skorlu model kazanır
        tahmin = max(skorlar, key=skorlar.get)
        sonuc = "✅" if tahmin == gercek_kelime else "❌"
        
        print(f"  Gerçek: {gercek_kelime:10} → Tahmin: {tahmin:10} {sonuc}")
        
        if tahmin == gercek_kelime:
            dogru += 1
        toplam += 1

print(f"\nDoğruluk: {dogru}/{toplam} = %{dogru/toplam*100:.1f}")
