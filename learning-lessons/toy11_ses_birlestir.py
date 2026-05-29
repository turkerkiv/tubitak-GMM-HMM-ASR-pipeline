import os
import random
from pydub import AudioSegment

VERI_KLASORU = "database"
CIKTI_KLASORU = "database_cumle"
KELIMELER = ["baslat", "geri", "yukari"]

os.makedirs(CIKTI_KLASORU, exist_ok=True)

# Her kelimeden wav listesi al
wav_listesi = {}
for kelime in KELIMELER:
    klasor = os.path.join(VERI_KLASORU, kelime)
    dosyalar = sorted([f for f in os.listdir(klasor) if f.endswith(".wav")])
    wav_listesi[kelime] = [os.path.join(klasor, d) for d in dosyalar]

# 20 farklı kelime kombinasyonu oluştur
random.seed(42)
kombinasyonlar = []

for i in range(20):
    # 2 veya 3 kelimelik rastgele kombinasyon
    uzunluk = random.choice([2, 3])
    kelimeler = [random.choice(KELIMELER) for _ in range(uzunluk)]
    kombinasyonlar.append(kelimeler)

print("Ses dosyaları birleştiriliyor...")

etiketler = []  # hangi dosya hangi kelimeler

for i, kelimeler in enumerate(kombinasyonlar):
    birlesik = AudioSegment.empty()
    
    for kelime in kelimeler:
        # O kelimeden rastgele bir wav seç
        wav_yolu = random.choice(wav_listesi[kelime])
        ses = AudioSegment.from_wav(wav_yolu)
        
        # Kelimeler arası 200ms sessizlik ekle
        sessizlik = AudioSegment.silent(duration=200)
        birlesik += ses + sessizlik
    
    # Kaydet
    dosya_adi = f"cumle_{i:02d}.wav"
    cikti_yolu = os.path.join(CIKTI_KLASORU, dosya_adi)
    birlesik.export(cikti_yolu, format="wav")
    
    etiket = " ".join(kelimeler)
    etiketler.append((dosya_adi, etiket))
    print(f"  {dosya_adi}: {etiket}")

# Etiketleri kaydet
with open(os.path.join(CIKTI_KLASORU, "etiketler.txt"), "w", encoding="utf-8") as f:
    for dosya, etiket in etiketler:
        f.write(f"{dosya}\t{etiket}\n")

print(f"\n{len(kombinasyonlar)} cümle oluşturuldu!")
print("database_cumle/ klasörüne kaydedildi.")
