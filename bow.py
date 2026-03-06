import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Load cleaned dataset from Step 1
df = pd.read_csv("sentiment_cleaned.csv")
text = df["clean_text"].values

# BoW
bow = CountVectorizer()
X_bow = bow.fit_transform(text)   # sparse matrix

print("✅ BoW created")
print("BoW shape (samples, vocab_size):", X_bow.shape)

# Show vocab size + first 25 vocab words
vocab = bow.get_feature_names_out()
print("\nVocabulary size:", len(vocab))
print("First 25 vocab words:", vocab[:len(vocab)])

# Show BoW vector for first sample (non-zero entries)
i = 0
row = X_bow[i].toarray()[0]
nonzero_idx = row.nonzero()[0]

print("\n🔹 Example text (clean_text[0]):")
print(df.loc[i, "clean_text"])

print("\n🔹 Non-zero BoW entries (word -> count):")
for idx in nonzero_idx:
    print(f"{vocab[idx]} -> {int(row[idx])}")

# ============================================================================
# COMPLETE BOW MATRIX
# ============================================================================
print("\n" + "="*80)
print("COMPLETE BAG OF WORDS MATRIX")
print("="*80)

# Convert to DataFrame for better visualization
bow_matrix_df = pd.DataFrame(
    X_bow.toarray(),
    columns=vocab,
    index=[f"Sample_{i+1}" for i in range(len(text))]
)

print("\nFull BoW Matrix (rows=samples, columns=words):\n")
print(bow_matrix_df)

print("\n" + "="*80)
print(f"Matrix Shape: {bow_matrix_df.shape[0]} samples × {bow_matrix_df.shape[1]} words")
print("="*80)