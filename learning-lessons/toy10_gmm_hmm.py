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
# FARK: GaussianHMM → GMMHMM
# GaussianHMM: her durum için tek Gaussian
# GMMHMM:      her durum için birden fazla Gaussian karışımı
#
# Yani her fonem durumu artık daha zengin temsil edilir
# "a" fonemi farklı kişilerde farklı çıkabilir
# GMM bu varyasyonu yakalar
# ---------------------------------------------

def veri_yukle():
    kelime_veri = defaultdict(list)
    
    for klasor_adi, kelime in KELIMELER.items():
        wav_klasoru = f"database/{klasor_adi}"
        tg_klasoru  = f"mfa_output/{klasor_adi}"
        
        wav_dosyalar = sorted([f for f in os.listdir(wav_klasoru) 
                               if f.endswith(".wav")])
        
        for wav_dosya in wav_dosyalar:
            wav_yolu = os.path.join(wav_klasoru, wav_dosya)
            tg_yolu  = os.path.join(tg_klasoru, 
                                    wav_dosya.replace(".wav", ".TextGrid"))
            
            if not os.path.exists(tg_yolu):
                continue
            
            y, sr = librosa.load(wav_yolu, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T
            
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
    
    return kelime_veri

def model_egit(ornekler, n_durum, n_mix):
    """
    n_mix = kaç Gaussian karışımı kullanılsın
    n_mix=1  → GaussianHMM ile aynı
    n_mix=3  → her durum için 3 Gaussian karışımı
    """
    model = hmm.GMMHMM(
        n_components=n_durum,
        n_mix=n_mix,
        covariance_type="diag",
        n_iter=N_ITER
    )
    
    egitim = ornekler[:7]
    X = np.concatenate([o["mfcc"] for o in egitim])
    lengths = [len(o["mfcc"]) for o in egitim]
    model.fit(X, lengths)
    return model

def test_et(modeller, kelime_veri):
    dogru = 0
    toplam = 0
    for gercek_kelime, ornekler in kelime_veri.items():
        for ornek in ornekler[7:]:
            skorlar = {k: m.score(ornek["mfcc"]) 
                      for k, m in modeller.items()}
            tahmin = max(skorlar, key=skorlar.get)
            sonuc = "✅" if tahmin == gercek_kelime else "❌"
            print(f"  Gerçek: {gercek_kelime:10} → "
                  f"Tahmin: {tahmin:10} {sonuc}")
            if tahmin == gercek_kelime:
                dogru += 1
            toplam += 1
    return dogru, toplam

# ---------------------------------------------
# YÜKLE
# ---------------------------------------------
print("Veri yükleniyor...")
kelime_veri = veri_yukle()
for k, v in kelime_veri.items():
    print(f"  {k}: {len(v)} dosya")

# ---------------------------------------------
# FARKLI N_MIX DEĞERLERİ İLE KARŞILAŞTIR
# ---------------------------------------------
print("\n" + "="*50)
print("        GMM-HMM KARŞILAŞTIRMASI")
print("="*50)

for n_mix in [1, 2, 3]:
    print(f"\n[n_mix={n_mix}] Her durum için {n_mix} Gaussian:")
    
    modeller = {}
    for kelime, ornekler in kelime_veri.items():
        n_durum = len(ornekler[0]["fonemler"])
        modeller[kelime] = model_egit(ornekler, n_durum, n_mix)
    
    dogru, toplam = test_et(modeller, kelime_veri)
    print(f"  → Doğruluk: {dogru}/{toplam} = %{dogru/toplam*100:.1f}")

print("\n" + "="*50)
print("ÖZET:")
print("  n_mix=1 → GaussianHMM ile aynı")
print("  n_mix=2 → Her fonem için 2 Gaussian")  
print("  n_mix=3 → Her fonem için 3 Gaussian")
print("  Daha fazla mix = daha zengin model")
print("  Ama çok fazla mix + az veri = overfitting")
