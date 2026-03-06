import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load cleaned dataset
df = pd.read_csv("sentiment_cleaned.csv")
text = df["clean_text"].values

# TF-IDF
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(text)  # sparse matrix

print("✅ TF-IDF created")
print("TF-IDF shape (samples, vocab_size):", X_tfidf.shape)

# Show vocab size + first 25 vocab words
vocab = tfidf.get_feature_names_out()
print("\nVocabulary size:", len(vocab))
print("First 25 vocab words:", vocab[:25])

# Show TF-IDF values for first sample (top non-zero entries)
i = 0
row = X_tfidf[i].toarray()[0]
nonzero_idx = row.nonzero()[0]

# Sort by TF-IDF value (descending)
pairs = sorted([(vocab[idx], float(row[idx])) for idx in nonzero_idx],
               key=lambda x: x[1], reverse=True)

print("\n🔹 Example text (clean_text[0]):")
print(df.loc[i, "clean_text"])

print("\n🔹 TF-IDF non-zero entries (word -> tfidf):")
for w, v in pairs:
    print(f"{w} -> {v:.4f}")

# ============================================================================
# COMPLETE TF-IDF MATRIX
# ============================================================================
print("\n" + "="*80)
print("COMPLETE TF-IDF MATRIX")
print("="*80)

# Convert to DataFrame for better visualization
tfidf_matrix_df = pd.DataFrame(
    X_tfidf.toarray(),
    columns=vocab,
    index=[f"Sample_{i+1}" for i in range(len(text))]
)

print("\nFull TF-IDF Matrix (rows=samples, columns=words):\n")
print(tfidf_matrix_df)

print("\n" + "="*80)
print(f"Matrix Shape: {tfidf_matrix_df.shape[0]} samples × {tfidf_matrix_df.shape[1]} words")
print("="*80)

# Show statistics
print("\nMatrix Statistics:")
print(f"• Min TF-IDF value: {tfidf_matrix_df.values.min():.4f}")
print(f"• Max TF-IDF value: {tfidf_matrix_df.values.max():.4f}")
print(f"• Average TF-IDF (non-zero): {tfidf_matrix_df.values[tfidf_matrix_df.values > 0].mean():.4f}")
print(f"• Sparsity: {(tfidf_matrix_df.values == 0).sum() / tfidf_matrix_df.size * 100:.2f}% zeros")
print("="*80)