import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack

# =========================
# 1) Load cleaned dataset
# =========================
df = pd.read_csv("sentiment_cleaned.csv")
text = df["clean_text"].values
Y = df["label"].values   # labels

# =========================
# 2) Rebuild features (BoW + TF-IDF) and concatenate
#    (do this again so this file is independent)
# =========================
bow = CountVectorizer()
X_bow = bow.fit_transform(text)

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(text)

combined_features = hstack([X_bow, X_tfidf])

print("✅ Combined features ready:", combined_features.shape)

# =========================
# 3) Train-test split (slide step)
# =========================
train_features, test_features, train_labels, test_labels = train_test_split(
    combined_features,
    Y,
    test_size=0.20,
    random_state=42,
    stratify=Y
)

print("\n✅ Split done")
print("Train features shape:", train_features.shape)
print("Test features shape :", test_features.shape)

# Optional: label distribution check
print("\n🔹 Train label counts:\n", pd.Series(train_labels).value_counts())
print("\n🔹 Test label counts:\n", pd.Series(test_labels).value_counts())

# Optional: show which sample indices went to test (useful for debugging)
# We re-split again using indices (same random_state, stratify)
idx = list(range(len(Y)))
idx_train, idx_test, _, _ = train_test_split(
    idx, Y, test_size=0.20, random_state=42, stratify=Y
)
print("\nTest sample indices:", idx_test)
print("Test samples (original text):")
for i in idx_test:
    print(f"- [{df.loc[i,'label']}] {df.loc[i,'text']}")