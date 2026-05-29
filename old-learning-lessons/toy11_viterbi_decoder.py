import os
import numpy as np
import librosa
from hmmlearn import hmm
import textgrid
from collections import defaultdict

KELIMELER = {
    "baslat": "başlat",
    "geri": "geri",
    "yukari": "yukarı"
}
N_MFCC = 13
N_ITER = 100
N_MIX = 3

# ---------------------------------------------
# ADIM 1: MODELLERİ EĞİT (toy10 ile aynı)
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
            if fonemler:
                kelime_veri[kelime].append({"mfcc": mfcc, "fonemler": fonemler})
    return kelime_veri

print("Modeller eğitiliyor...")
kelime_veri = veri_yukle()
modeller = {}
for kelime, ornekler in kelime_veri.items():
    n_durum = len(ornekler[0]["fonemler"])
    model = hmm.GMMHMM(
        n_components=n_durum,
        n_mix=N_MIX,
        covariance_type="diag",
        n_iter=N_ITER
    )
    X = np.concatenate([o["mfcc"] for o in ornekler])
    lengths = [len(o["mfcc"]) for o in ornekler]
    model.fit(X, lengths)
    modeller[kelime] = model
    print(f"  {kelime} eğitildi")

# ---------------------------------------------
# ADIM 2: VİTERBİ DECODER
#
# Fikir şu:
# Gelen ses → MFCC çıkar
# Sesi küçük pencerelere böl (her pencere ~1 kelime uzunluğu)
# Her pencereyi tüm modellere sor
# En yüksek skor → o penceredeki kelime
# Tüm pencerelerin kelimelerini birleştir → cümle
# ---------------------------------------------

def viterbi_decoder(mfcc, pencere_boyutu=50, adim=25):
    """
    mfcc         → (zaman, 13) matris
    pencere_boyutu → kaç frame'lik pencere
    adim         → pencere kaçar kaçar kayacak
    
    Sliding window yaklaşımı:
    |--pencere--|
        |--pencere--|
            |--pencere--|
    """
    tahminler = []
    skorlar_listesi = []
    
    for baslangic in range(0, len(mfcc) - pencere_boyutu, adim):
        bitis = baslangic + pencere_boyutu
        pencere = mfcc[baslangic:bitis]
        
        # Her modele sor
        skorlar = {}
        for kelime, model in modeller.items():
            try:
                skorlar[kelime] = model.score(pencere)
            except:
                skorlar[kelime] = float('-inf')
        
        en_iyi = max(skorlar, key=skorlar.get)
        tahminler.append(en_iyi)
        skorlar_listesi.append(skorlar)
    
    return tahminler

def tekrarlari_temizle(tahminler):
    """
    ['başlat', 'başlat', 'geri', 'geri', 'geri', 'yukarı']
    →
    ['başlat', 'geri', 'yukarı']
    """
    if not tahminler:
        return []
    temiz = [tahminler[0]]
    for t in tahminler[1:]:
        if t != temiz[-1]:
            temiz.append(t)
    return temiz

# ---------------------------------------------
# ADIM 3: CÜMLE DOSYALARINI TEST ET
# ---------------------------------------------
print("\n--- CÜMLE TANIMA TEST SONUÇLARI ---\n")

etiket_dosyasi = "database_cumle/etiketler.txt"
etiketler = {}
with open(etiket_dosyasi, "r", encoding="utf-8") as f:
    for satir in f:
        dosya, etiket = satir.strip().split("\t")
        # Türkçe karakterlere çevir
        etiket = (etiket
                  .replace("baslat", "başlat")
                  .replace("yukari", "yukarı"))
        etiketler[dosya] = etiket

dogru = 0
toplam = 0

for dosya, gercek_etiket in sorted(etiketler.items()):
    wav_yolu = os.path.join("database_cumle", dosya)
    y, sr = librosa.load(wav_yolu, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T
    
    # Viterbi decoder çalıştır
    tahminler = viterbi_decoder(mfcc)
    temiz_tahmin = tekrarlari_temizle(tahminler)
    tahmin_str = " ".join(temiz_tahmin)
    
    sonuc = "✅" if tahmin_str == gercek_etiket else "❌"
    print(f"{sonuc} Gerçek : {gercek_etiket}")
    print(f"   Tahmin: {tahmin_str}\n")
    
    if tahmin_str == gercek_etiket:
        dogru += 1
    toplam += 1

print(f"Doğruluk: {dogru}/{toplam} = %{dogru/toplam*100:.1f}")
