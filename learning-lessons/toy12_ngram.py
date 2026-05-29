import os
import numpy as np
import librosa
from hmmlearn import hmm
import textgrid
from collections import defaultdict, Counter
import nltk

KELIMELER = {
    "baslat": "başlat",
    "geri": "geri",
    "yukari": "yukarı"
}
N_MFCC = 13
N_ITER = 100
N_MIX = 3

# ---------------------------------------------
# ADIM 1: MODELLERİ EĞİT (toy11 ile aynı)
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
# ADIM 2: N-GRAM DİL MODELİ EĞİT
#
# Eğitim cümlelerinden bigram olasılıkları çıkar
# "başlat geri" → P(geri|başlat) = ?
# "geri yukarı" → P(yukarı|geri) = ?
# ---------------------------------------------
print("\nN-gram modeli eğitiliyor...")

# Etiket dosyasından cümleleri oku
egitim_cumleleri = []
with open("database_cumle/etiketler.txt", "r", encoding="utf-8") as f:
    for satir in f:
        _, etiket = satir.strip().split("\t")
        etiket = (etiket
                  .replace("baslat", "başlat")
                  .replace("yukari", "yukarı"))
        kelimeler = etiket.split()
        egitim_cumleleri.append(kelimeler)

# Bigram sayılarını hesapla
unigram_sayac = Counter()
bigram_sayac = Counter()

for cumle in egitim_cumleleri:
    for kelime in cumle:
        unigram_sayac[kelime] += 1
    for i in range(len(cumle) - 1):
        bigram = (cumle[i], cumle[i+1])
        bigram_sayac[bigram] += 1

def bigram_olasiligi(onceki, sonraki, alpha=0.1):
    """
    P(sonraki | onceki) hesapla
    alpha = Laplace smoothing (görülmemiş bigram için)
    """
    pay = bigram_sayac[(onceki, sonraki)] + alpha
    payda = unigram_sayac[onceki] + alpha * len(KELIMELER)
    return np.log(pay / payda)  # log olasılık

print("  Bigram sayıları:")
kelime_listesi = list(modeller.keys())
for k1 in kelime_listesi:
    for k2 in kelime_listesi:
        sayi = bigram_sayac[(k1, k2)]
        print(f"    P({k2}|{k1}) → {sayi} kez görüldü")

# ---------------------------------------------
# ADIM 3: VİTERBİ + N-GRAM DECODER
#
# Artık sadece akustik skora değil
# bigram olasılığını da hesaba katıyoruz
#
# Toplam skor = Akustik skor + N-gram skoru
# ---------------------------------------------

def viterbi_ngram_decoder(mfcc, pencere_boyutu=50, adim=25, ngram_agirlik=0.5):
    """
    Her pencere için:
    1. Akustik skor hesapla (HMM)
    2. Bir önceki kelimeye göre N-gram skoru ekle
    3. İkisini birleştir
    """
    tahminler = []
    onceki_kelime = None
    
    for baslangic in range(0, len(mfcc) - pencere_boyutu, adim):
        bitis = baslangic + pencere_boyutu
        pencere = mfcc[baslangic:bitis]
        
        skorlar = {}
        for kelime, model in modeller.items():
            try:
                # Akustik skor
                akustik = model.score(pencere)
                
                # N-gram skoru
                if onceki_kelime is not None:
                    ngram = bigram_olasiligi(onceki_kelime, kelime)
                else:
                    # İlk kelime için unigram
                    ngram = np.log((unigram_sayac[kelime] + 0.1) / 
                                  (sum(unigram_sayac.values()) + 0.1))
                
                # Toplam skor
                skorlar[kelime] = akustik + ngram_agirlik * ngram
                
            except:
                skorlar[kelime] = float('-inf')
        
        en_iyi = max(skorlar, key=skorlar.get)
        tahminler.append(en_iyi)
        onceki_kelime = en_iyi
    
    return tahminler

def tekrarlari_temizle(tahminler):
    if not tahminler:
        return []
    temiz = [tahminler[0]]
    for t in tahminler[1:]:
        if t != temiz[-1]:
            temiz.append(t)
    return temiz

# ---------------------------------------------
# ADIM 4: TEST ET VE TOY11 İLE KARŞILAŞTIR
# ---------------------------------------------
print("\n--- TOY11 vs TOY12 KARŞILAŞTIRMA ---\n")

etiketler = {}
with open("database_cumle/etiketler.txt", "r", encoding="utf-8") as f:
    for satir in f:
        dosya, etiket = satir.strip().split("\t")
        etiket = (etiket
                  .replace("baslat", "başlat")
                  .replace("yukari", "yukarı"))
        etiketler[dosya] = etiket

dogru_ngram = 0
toplam = 0

for dosya, gercek_etiket in sorted(etiketler.items()):
    wav_yolu = os.path.join("database_cumle", dosya)
    y, sr = librosa.load(wav_yolu, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T
    
    tahminler = viterbi_ngram_decoder(mfcc)
    temiz = tekrarlari_temizle(tahminler)
    tahmin_str = " ".join(temiz)
    
    sonuc = "✅" if tahmin_str == gercek_etiket else "❌"
    print(f"{sonuc} Gerçek : {gercek_etiket}")
    print(f"   Tahmin: {tahmin_str}\n")
    
    if tahmin_str == gercek_etiket:
        dogru_ngram += 1
    toplam += 1

print("="*45)
print(f"Toy11 (Viterbi)         : %5.0")
print(f"Toy12 (Viterbi + N-gram): %{dogru_ngram/toplam*100:.1f}")
print("="*45)
