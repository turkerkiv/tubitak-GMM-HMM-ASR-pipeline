from pocketsphinx import LiveSpeech, get_model_path

model_path = get_model_path()
print(f"Model yolu: {model_path}")

# Mevcut modelleri listele
import os
print("\nMevcut modeller:")
for f in os.listdir(model_path):
    print(f"  {f}")
