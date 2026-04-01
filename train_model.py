import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from preprocess import clean_text

# Sample dataset (replace with real CSV later)
data = {
    "tweet": [
        "I love this product",
        "This is terrible",
        "It's okay",
        "Amazing experience",
        "Worst ever"
    ],
    "sentiment": [
        "positive", "negative", "neutral", "positive", "negative"
    ]
}

df = pd.DataFrame(data)

# Clean text
df['cleaned'] = df['tweet'].apply(clean_text)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned'])
y = df['sentiment']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model & vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained and saved!")
