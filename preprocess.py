import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

# Download required NLTK data (run once)
try:
    nltk.data.find('corpora/wordnet.zip')
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger_eng')

# Load dataset
df = pd.read_csv("sentiment.csv")

# Stop words to remove (common words with little meaning)
STOP_WORDS = ["am", "is", "are", "a", "an", "the", "and", "or", "to", "of", 
              "in", "on", "for", "this", "that", "it", "as", "at", "be", "by"]

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Convert NLTK POS tags to WordNet POS tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ      # Adjective
    elif tag.startswith('V'):
        return wordnet.VERB     # Verb
    elif tag.startswith('N'):
        return wordnet.NOUN     # Noun
    elif tag.startswith('R'):
        return wordnet.ADV      # Adverb
    else:
        return wordnet.NOUN     # Default to noun

# ALL-IN-ONE preprocessing function (all steps combined)
def preprocess_text(text):
    # Step 1: Convert to lowercase
    text = str(text).lower()
    
    # Step 2: Remove punctuation (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Step 3: Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Step 4: Remove stop words
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]
    
    # Step 5: POS tagging and Lemmatization with correct POS
    # This properly converts: disappointed→disappoint, running→run, better→good
    pos_tags = nltk.pos_tag(words)
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) 
                  for word, pos in pos_tags]
    
    return ' '.join(lemmatized)

# Apply preprocessing
df["clean_text"] = df["text"].apply(preprocess_text)

# Show results
print("✅ Text Preprocessing Complete (with Lemmatization)\n")
print("Original → Cleaned (first 10 examples):\n")
for i in range(min(10, len(df))):
    print(f"ORIGINAL: {df.loc[i, 'text']}")
    print(f"CLEANED : {df.loc[i, 'clean_text']}\n")

# Save cleaned data
df.to_csv("sentiment_cleaned.csv", index=False)
print("✅ Saved: sentiment_cleaned.csv")