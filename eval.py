import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, cohen_kappa_score)

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
# 4) Train model + predict
# =========================
lr_clf = LogisticRegression(max_iter=300)
lr_clf.fit(train_features, train_labels)
test_pred = lr_clf.predict(test_features)

# =========================
# 5) Evaluation (slide)
# =========================
print("=== Confusion Table (pd.crosstab) ===")
confusion_table = pd.crosstab(
    pd.Series(test_labels, name="Actual"),
    pd.Series(test_pred, name="Predicted")
)
print(confusion_table)

print("\n=== Classification Report ===")
print(classification_report(test_labels, test_pred))

# =========================
# 6) COMPREHENSIVE CONFUSION MATRIX ANALYSIS
# =========================
print("\n" + "="*80)
print("COMPREHENSIVE CONFUSION MATRIX ANALYSIS")
print("="*80)

# Get unique labels
labels = sorted(list(set(test_labels)))

# Calculate confusion matrix
cm = confusion_matrix(test_labels, test_pred, labels=labels)

# Calculate normalized confusion matrix (percentages)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Calculate metrics
accuracy = accuracy_score(test_labels, test_pred)
precision = precision_score(test_labels, test_pred, average='weighted', zero_division=0)
recall = recall_score(test_labels, test_pred, average='weighted', zero_division=0)
f1 = f1_score(test_labels, test_pred, average='weighted', zero_division=0)
kappa = cohen_kappa_score(test_labels, test_pred)

# Print detailed metrics
print("\n📊 OVERALL METRICS:")
print(f"  • Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  • Precision: {precision:.4f}")
print(f"  • Recall:    {recall:.4f}")
print(f"  • F1-Score:  {f1:.4f}")
print(f"  • Cohen's Kappa: {kappa:.4f}")

# Per-class metrics
print("\n📊 PER-CLASS METRICS:")
for i, label in enumerate(labels):
    class_precision = precision_score(test_labels, test_pred, labels=[label], average='micro', zero_division=0)
    class_recall = recall_score(test_labels, test_pred, labels=[label], average='micro', zero_division=0)
    class_f1 = f1_score(test_labels, test_pred, labels=[label], average='micro', zero_division=0)
    support = (test_labels == label).sum()
    print(f"\n  Class: {label}")
    print(f"    - Precision: {class_precision:.4f}")
    print(f"    - Recall:    {class_recall:.4f}")
    print(f"    - F1-Score:  {class_f1:.4f}")
    print(f"    - Support:   {support}")

# =========================
# 7) VISUALIZATIONS
# =========================
print("\n" + "="*80)
print("Generating confusion matrix visualizations...")
print("="*80)

fig = plt.figure(figsize=(16, 12))

# 1. Confusion Matrix (Counts)
ax1 = plt.subplot(2, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels,
            cbar_kws={'label': 'Count'}, ax=ax1)
ax1.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
ax1.set_ylabel('Actual Label', fontsize=11, fontweight='bold')
ax1.set_title('Confusion Matrix (Counts)', fontsize=13, fontweight='bold')

# 2. Normalized Confusion Matrix (Percentages)
ax2 = plt.subplot(2, 3, 2)
sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Greens',
            xticklabels=labels, yticklabels=labels,
            cbar_kws={'label': 'Percentage (%)'}, ax=ax2)
ax2.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
ax2.set_ylabel('Actual Label', fontsize=11, fontweight='bold')
ax2.set_title('Confusion Matrix (Normalized %)', fontsize=13, fontweight='bold')

# 3. Per-Class Metrics Bar Chart
ax3 = plt.subplot(2, 3, 3)
class_metrics = []
for label in labels:
    p = precision_score(test_labels, test_pred, labels=[label], average='micro', zero_division=0)
    r = recall_score(test_labels, test_pred, labels=[label], average='micro', zero_division=0)
    f = f1_score(test_labels, test_pred, labels=[label], average='micro', zero_division=0)
    class_metrics.append({'Class': label, 'Precision': p, 'Recall': r, 'F1-Score': f})

df_metrics = pd.DataFrame(class_metrics)
x = np.arange(len(labels))
width = 0.25
ax3.bar(x - width, df_metrics['Precision'], width, label='Precision', alpha=0.8)
ax3.bar(x, df_metrics['Recall'], width, label='Recall', alpha=0.8)
ax3.bar(x + width, df_metrics['F1-Score'], width, label='F1-Score', alpha=0.8)
ax3.set_xlabel('Class', fontsize=11, fontweight='bold')
ax3.set_ylabel('Score', fontsize=11, fontweight='bold')
ax3.set_title('Per-Class Metrics', fontsize=13, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(labels)
ax3.legend()
ax3.set_ylim(0, 1.1)
ax3.grid(axis='y', alpha=0.3)

# 4. Overall Metrics Summary
ax4 = plt.subplot(2, 3, 4)
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [accuracy, precision, recall, f1]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = ax4.barh(metrics_names, metrics_values, color=colors, alpha=0.7)
ax4.set_xlabel('Score', fontsize=11, fontweight='bold')
ax4.set_title('Overall Model Performance', fontsize=13, fontweight='bold')
ax4.set_xlim(0, 1.1)
ax4.grid(axis='x', alpha=0.3)
# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, metrics_values)):
    ax4.text(value + 0.02, i, f'{value:.3f}', va='center', fontweight='bold')

# 5. True vs Predicted Comparison
ax5 = plt.subplot(2, 3, 5)
sample_indices = np.arange(min(15, len(test_labels)))
ax5.scatter(sample_indices, [labels.index(l) for l in test_labels[:15]], 
            label='Actual', marker='o', s=100, alpha=0.6)
ax5.scatter(sample_indices, [labels.index(l) for l in test_pred[:15]], 
            label='Predicted', marker='x', s=100, alpha=0.6)
ax5.set_xlabel('Sample Index', fontsize=11, fontweight='bold')
ax5.set_ylabel('Class', fontsize=11, fontweight='bold')
ax5.set_yticks(range(len(labels)))
ax5.set_yticklabels(labels)
ax5.set_title('Actual vs Predicted (First 15 Samples)', fontsize=13, fontweight='bold')
ax5.legend()
ax5.grid(alpha=0.3)

# 6. Class Distribution
ax6 = plt.subplot(2, 3, 6)
actual_dist = pd.Series(test_labels).value_counts()
pred_dist = pd.Series(test_pred).value_counts()
x = np.arange(len(labels))
width = 0.35
ax6.bar(x - width/2, [actual_dist.get(l, 0) for l in labels], 
        width, label='Actual', alpha=0.8)
ax6.bar(x + width/2, [pred_dist.get(l, 0) for l in labels], 
        width, label='Predicted', alpha=0.8)
ax6.set_xlabel('Class', fontsize=11, fontweight='bold')
ax6.set_ylabel('Count', fontsize=11, fontweight='bold')
ax6.set_title('Class Distribution (Test Set)', fontsize=13, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(labels)
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

plt.suptitle('Complete Confusion Matrix Analysis', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('confusion_matrix_complete.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualization saved as 'confusion_matrix_complete.png'")
plt.show()

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)