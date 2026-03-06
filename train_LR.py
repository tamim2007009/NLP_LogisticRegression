import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression

# =========================
# 1) Load cleaned dataset
# =========================
df = pd.read_csv("sentiment_cleaned.csv")
text = df["clean_text"].values
Y = df["label"].values

# =========================
# 2) Feature extraction + concatenation
# =========================
bow = CountVectorizer()
X_bow = bow.fit_transform(text)

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(text)

combined_features = hstack([X_bow, X_tfidf])

# =========================
# 3) Train-test split
# =========================
train_features, test_features, train_labels, test_labels = train_test_split(
    combined_features, Y, test_size=0.20, random_state=42, stratify=Y
)

# =========================
# 4) Train Logistic Regression
# =========================
lr_clf = LogisticRegression(max_iter=300)
lr_clf.fit(train_features, train_labels)

print("✅ Logistic Regression trained")

# =========================
# 5) Predict on test set
# =========================
test_pred = lr_clf.predict(test_features)

print("\n✅ Predictions done")
print("Test labels :", list(test_labels))
print("Predicted   :", list(test_pred))