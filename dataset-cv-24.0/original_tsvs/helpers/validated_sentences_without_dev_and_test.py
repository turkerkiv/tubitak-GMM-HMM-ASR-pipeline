import pandas as pd

# Dosyaları oku
train = pd.read_csv("train.tsv", sep="\t")
test = pd.read_csv("test.tsv", sep="\t")
dev = pd.read_csv("dev.tsv", sep="\t")
validated = pd.read_csv("validated_sentences.tsv", sep="\t")

# sentence_id setleri
test_ids = set(test["sentence_id"].dropna().astype(str).str.strip())
dev_ids = set(dev["sentence_id"].dropna().astype(str).str.strip())

# Çıkarmak istediğin tüm id'ler
remove_ids = test_ids | dev_ids

# validated içinden bunları çıkar
new_validated = validated[
    ~validated["sentence_id"].astype(str).str.strip().isin(remove_ids)
]

# Yeni dosyayı kaydet
new_validated.to_csv(
    "validated_sentences_filtered.tsv",
    sep="\t",
    index=False
)

print("Eski validated sayısı:", len(validated))
print("Yeni validated sayısı:", len(new_validated))
print("Çıkarılan satır sayısı:", len(validated) - len(new_validated))

# Kontrol amaçlı
remaining_overlap_test = set(new_validated["sentence_id"]) & test_ids
remaining_overlap_dev = set(new_validated["sentence_id"]) & dev_ids

print("Kalan Test overlap:", len(remaining_overlap_test))
print("Kalan Dev overlap:", len(remaining_overlap_dev))