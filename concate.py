import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack

# Load cleaned dataset
df = pd.read_csv("sentiment_cleaned.csv")
text = df["clean_text"].values

# 1) BoW
bow = CountVectorizer()
X_bow = bow.fit_transform(text)

# 2) TF-IDF
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(text)

# 3) Concatenate (sparse-safe)
combined_features = hstack([X_bow, X_tfidf])

print("✅ Feature concatenation done")
print("BoW shape     :", X_bow.shape)
print("TF-IDF shape  :", X_tfidf.shape)
print("Combined shape:", combined_features.shape)

# Quick sanity check: show how many non-zero entries in first row
nz_bow = X_bow[0].count_nonzero()
nz_tfidf = X_tfidf[0].count_nonzero()
nz_combo = combined_features[0].count_nonzero()
print("\nNon-zeros in row0 -> BoW:", nz_bow, "| TF-IDF:", nz_tfidf, "| Combined:", nz_combo)

# ============================================================================
# COMPLETE COMBINED MATRIX
# ============================================================================
print("\n" + "="*80)
print("COMPLETE COMBINED MATRIX (BoW + TF-IDF)")
print("="*80)

# Get feature names
bow_vocab = bow.get_feature_names_out()
tfidf_vocab = tfidf.get_feature_names_out()

# Create column names with prefixes to distinguish BoW from TF-IDF
bow_columns = [f"bow_{word}" for word in bow_vocab]
tfidf_columns = [f"tfidf_{word}" for word in tfidf_vocab]
all_columns = bow_columns + tfidf_columns

# Convert to DataFrame
combined_matrix_df = pd.DataFrame(
    combined_features.toarray(),
    columns=all_columns,
    index=[f"Sample_{i+1}" for i in range(len(text))]
)

print("\nFull Combined Matrix (rows=samples, columns=bow_features + tfidf_features):\n")
print(combined_matrix_df)

print("\n" + "="*80)
print(f"Matrix Shape: {combined_matrix_df.shape[0]} samples × {combined_matrix_df.shape[1]} features")
print(f"  • BoW features: {len(bow_columns)}")
print(f"  • TF-IDF features: {len(tfidf_columns)}")
print("="*80)

# Show statistics
print("\nMatrix Statistics:")
print(f"• Min value: {combined_matrix_df.values.min():.4f}")
print(f"• Max value: {combined_matrix_df.values.max():.4f}")
print(f"• Average value (non-zero): {combined_matrix_df.values[combined_matrix_df.values > 0].mean():.4f}")
print(f"• Sparsity: {(combined_matrix_df.values == 0).sum() / combined_matrix_df.size * 100:.2f}% zeros")
print("="*80)