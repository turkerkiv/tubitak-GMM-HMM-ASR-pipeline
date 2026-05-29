import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt

# ---------------------------------------------
# SENARYO:
# 2 farklı "kelime" var: "evet" ve "hayır"
# Her kelimeyi temsil eden MFCC benzeri sayılar üretiyoruz
# HMM bunları ayırt etmeyi öğrenecek
# ---------------------------------------------

np.random.seed(42)

def kelime_olustur(ortalama, sayi=20):
    """
    Gerçekte burada MFCC vektörleri olurdu.
    Şimdilik basit sayılar üretiyoruz.
    ortalama=1.0 → 'evet' sesi
    ortalama=5.0 → 'hayir' sesi
    """
    return np.random.normal(ortalama, 0.5, (sayi, 1))

# Eğitim verisi oluştur
# 10 tane "evet" örneği, 10 tane "hayir" örneği
print("Eğitim verisi oluşturuluyor...")

evet_ornekleri  = [kelime_olustur(1.0) for _ in range(10)]
hayir_ornekleri = [kelime_olustur(5.0) for _ in range(10)]

# HMM modeli oluştur
# n_components = kaç gizli durum olsun (fonem sayısı gibi düşün)
# covariance_type = 'diag' → hafif ve hızlı
print("HMM modelleri eğitiliyor...")

evet_modeli  = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100)
hayir_modeli = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100)

# Eğitim için verileri birleştir
evet_data  = np.concatenate(evet_ornekleri)
hayir_data = np.concatenate(hayir_ornekleri)

evet_modeli.fit(evet_data)
hayir_modeli.fit(hayir_data)

print("Eğitim tamamlandı!\n")

# ---------------------------------------------
# TEST: Yeni bir ses geldi, hangi kelime?
# ---------------------------------------------
print("--- TEST SONUÇLARI ---")

test_sesleri = {
    "evet gibi ses  (ort=1.0)": kelime_olustur(1.0, sayi=10),
    "hayir gibi ses (ort=5.0)": kelime_olustur(5.0, sayi=10),
    "belirsiz ses   (ort=3.0)": kelime_olustur(3.0, sayi=10),
}

for ses_adi, test_verisi in test_sesleri.items():
    # Her model bu sesin kendine ait olma olasılığını hesaplar
    evet_skoru  = evet_modeli.score(test_verisi)
    hayir_skoru = hayir_modeli.score(test_verisi)
    
    tahmin = "EVET" if evet_skoru > hayir_skoru else "HAYIR"
    
    print(f"\nSes: {ses_adi}")
    print(f"  evet_modeli skoru : {evet_skoru:.2f}")
    print(f"  hayir_modeli skoru: {hayir_skoru:.2f}")
    print(f"  → Tahmin: {tahmin}")

# ---------------------------------------------
# GÖRSELLEŞTİRME
# ---------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(evet_data,  bins=20, alpha=0.7, color='green', label='evet verisi')
axes[0].hist(hayir_data, bins=20, alpha=0.7, color='red',   label='hayir verisi')
axes[0].set_title("Eğitim Verisi Dağılımı")
axes[0].legend()

for ses_adi, test_verisi in test_sesleri.items():
    axes[1].scatter(range(len(test_verisi)), test_verisi, 
                   label=ses_adi, alpha=0.7)
axes[1].set_title("Test Verileri")
axes[1].legend(fontsize=7)

plt.tight_layout()
plt.savefig("toy3_sonuc.png")
plt.show()

print("\nBitti! toy3_sonuc.png kaydedildi.")
