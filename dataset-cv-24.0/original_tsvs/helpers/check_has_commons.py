import pandas as pd

# Dosyaları oku
train = pd.read_csv("train.tsv", sep="\t")
test = pd.read_csv("test.tsv", sep="\t")
dev = pd.read_csv("dev.tsv", sep="\t")
validated = pd.read_csv("validated_sentences.tsv", sep="\t")

# Common Voice formatında genelde cümleler "sentence_id" kolonunda olur
train_sentences = set(train["sentence_id"].dropna().str.strip())
test_sentences = set(test["sentence_id"].dropna().str.strip())
dev_sentences = set(dev["sentence_id"].dropna().str.strip())
validated_sentences = set(validated["sentence_id"].dropna().str.strip())

# Kesişimleri bul
train_test = train_sentences & test_sentences
train_dev = train_sentences & dev_sentences
test_dev = test_sentences & dev_sentences
validated_dev = validated_sentences & dev_sentences
validated_train = validated_sentences & train_sentences
validated_test = validated_sentences & test_sentences

# Sonuçları yazdır
print(f"Train-Test ortak cümle sayısı: {len(train_test)}")
print(f"Train-Dev ortak cümle sayısı: {len(train_dev)}")
print(f"Test-Dev ortak cümle sayısı: {len(test_dev)}")
print(f"Validated-Dev ortak cümle sayısı: {len(validated_dev)}")
print(f"Validated-Train ortak cümle sayısı: {len(validated_train)}")
print(f"Validated-Test ortak cümle sayısı: {len(validated_test)}")
