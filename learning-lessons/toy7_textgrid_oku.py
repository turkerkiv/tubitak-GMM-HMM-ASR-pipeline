import os
import textgrid
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# pip install textgrid
# önce kur: pip install textgrid

# Bir dosyayı okuyalım
KELIME = "baslat"
wav_dosya = f"database/{KELIME}/{os.listdir(f'database/{KELIME}')[0]}"
tg_dosya = wav_dosya.replace(f"database/{KELIME}/", f"mfa_output/{KELIME}/").replace(".wav", ".TextGrid")

print(f"Ses dosyası : {wav_dosya}")
print(f"TextGrid    : {tg_dosya}")

# TextGrid oku
tg = textgrid.TextGrid.fromFile(tg_dosya)

print("\n--- FONEM HİZALAMASI ---")
for tier in tg.tiers:
    if tier.name == "phones":
        for interval in tier:
            if interval.mark.strip():
                print(f"  {interval.minTime:.3f}s - {interval.maxTime:.3f}s : '{interval.mark}'")

# Görselleştir
y, sr = librosa.load(wav_dosya, sr=None)

fig, axes = plt.subplots(2, 1, figsize=(12, 6))

# Ses dalgası
librosa.display.waveshow(y, sr=sr, ax=axes[0])
axes[0].set_title(f"'{KELIME}' - Ses Dalgası + Fonem Sınırları")

# Fonem sınırlarını çiz
renkler = plt.cm.Set3(np.linspace(0, 1, 20))
i = 0
for tier in tg.tiers:
    if tier.name == "phones":
        for interval in tier:
            if interval.mark.strip():
                # dikey çizgi
                axes[0].axvline(x=interval.minTime, color='red', alpha=0.7, linewidth=1)
                axes[0].axvline(x=interval.maxTime, color='red', alpha=0.7, linewidth=1)
                # fonem etiketi
                orta = (interval.minTime + interval.maxTime) / 2
                axes[0].text(orta, 0.4, interval.mark, 
                           fontsize=12, ha='center', fontweight='bold', color='darkred')
                i += 1

# MFCC
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=axes[1])
axes[1].set_title("MFCC")

# MFCC üzerine de fonem sınırları
for tier in tg.tiers:
    if tier.name == "phones":
        for interval in tier:
            if interval.mark.strip():
                axes[1].axvline(x=interval.minTime, color='white', alpha=0.7, linewidth=1)

plt.tight_layout()
plt.savefig("toy7_sonuc.png")
plt.show()
print("\nBitti! toy7_sonuc.png kaydedildi.")
