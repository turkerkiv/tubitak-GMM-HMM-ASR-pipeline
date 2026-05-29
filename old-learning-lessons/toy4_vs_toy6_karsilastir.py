import os
import numpy as np
import librosa
from hmmlearn import hmm
from phonemizer import phonemize

VERI_KLASORU = "database"
KELIMELER = ["baslat", "geri", "yukari"]
N_MFCC = 13
N_ITER = 100

# ---------------------------------------------
# ORTAK FONKSİYONLAR
# ---------------------------------------------
def mfcc_cikar(dosya_yolu):
    y, sr = librosa.load(dosya_yolu, sr=None)
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T

def veri_yukle():
    veri = {}
    for kelime in KELIMELER:
        klasor = os.path.join(VERI_KLASORU, kelime)
        dosyalar = sorted([f for f in os.listdir(klasor) if f.endswith(".wav")])
        veri[kelime] = [mfcc_cikar(os.path.join(klasor, d)) for d in dosyalar]
    return veri

def test_et(modeller, veri):
    dogru = 0
    toplam = 0
    for gercek_kelime in KELIMELER:
        for test_mfcc in veri[gercek_kelime][7:]:
            skorlar = {k: m.score(test_mfcc) for k, m in modeller.items()}
            tahmin = max(skorlar, key=skorlar.get)
            if tahmin == gercek_kelime:
                dogru += 1
            toplam += 1
    return dogru, toplam

# ---------------------------------------------
# VERİ YÜKLE (İKİSİ DE AYNI VERİYİ KULLANIYOR)
# ---------------------------------------------
print("Veri yükleniyor...")
veri = veri_yukle()

# ---------------------------------------------
# TOY4: SABİT 3 GİZLİ DURUM
# ---------------------------------------------
print("\n[TOY4] Sabit 3 gizli durum ile eğitim...")
modeller_toy4 = {}
for kelime in KELIMELER:
    model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=N_ITER)
    X = np.concatenate(veri[kelime][:7])
    lengths = [len(m) for m in veri[kelime][:7]]
    model.fit(X, lengths)
    modeller_toy4[kelime] = model
    print(f"  {kelime}: 3 gizli durum")

dogru4, toplam4 = test_et(modeller_toy4, veri)

# ---------------------------------------------
# TOY6: FONEM SAYISI KADAR GİZLİ DURUM
# ---------------------------------------------
print("\n[TOY6] Fonem sayısı kadar gizli durum ile eğitim...")
kelime_fonem = {}
for kelime in KELIMELER:
    fonemler = list(phonemize(kelime, language='tr', backend='espeak').strip())
    kelime_fonem[kelime] = fonemler

modeller_toy6 = {}
for kelime in KELIMELER:
    n_durum = len(kelime_fonem[kelime])
    model = hmm.GaussianHMM(n_components=n_durum, covariance_type="diag", n_iter=N_ITER)
    X = np.concatenate(veri[kelime][:7])
    lengths = [len(m) for m in veri[kelime][:7]]
    model.fit(X, lengths)
    modeller_toy6[kelime] = model
    print(f"  {kelime}: {n_durum} gizli durum {kelime_fonem[kelime]}")

dogru6, toplam6 = test_et(modeller_toy6, veri)

# ---------------------------------------------
# KARŞILAŞTIRMA
# ---------------------------------------------
print("\n" + "="*45)
print("         KARŞILAŞTIRMA SONUCU")
print("="*45)
print(f"  TOY4 (sabit 3 durum)    : {dogru4}/{toplam4} = %{dogru4/toplam4*100:.1f}")
print(f"  TOY6 (fonem kadar durum): {dogru6}/{toplam6} = %{dogru6/toplam6*100:.1f}")
print("="*45)

if dogru6 > dogru4:
    print("  → TOY6 daha iyi! Fonem sayısı önemli.")
elif dogru6 == dogru4:
    print("  → İkisi eşit. Veri az olduğu için fark çıkmadı.")
else:
    print("  → TOY4 daha iyi çıktı. Az veriyle daha az durum bazen daha iyi.")
