import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# librosa'nın kendi örnek ses dosyasını kullanıyoruz
# kendi ses dosyan da olsa yolu yazabilirsin
audio_path = 'common_voice_tr_17341271.wav'

# Ses dosyasını yükle
# y = ses dalgası (sayı dizisi), sr = örnekleme frekansı (Hz)
y, sr = librosa.load(audio_path)

print(f"Ses süresi: {len(y)/sr:.2f} saniye")
print(f"Örnekleme frekansı: {sr} Hz")
print(f"Toplam örnek sayısı: {len(y)}")

# 3 grafik yan yana
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# 1. Grafik: Ham ses dalgası
librosa.display.waveshow(y, sr=sr, ax=axes[0])
axes[0].set_title("Ham Ses Dalgası")

# 2. Grafik: Spektrogram (sesin frekans içeriği zamana göre)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axes[1])
axes[1].set_title("Spektrogram")

# 3. Grafik: MFCC (bizim kullanacağımız öznitelikler)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=axes[2])
axes[2].set_title("MFCC Öznitelikleri (13 katsayı)")

plt.tight_layout()
plt.savefig("toy1_sonuc.png")
plt.show()

print("Bitti! toy1_sonuc.png kaydedildi.")
