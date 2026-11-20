import json
import joblib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# We use 4 categories only, so it's like routing tickets to 4 departments.
# (The interview story: "the dataset stands in for support tickets")
CATEGORIES = ['rec.sport.baseball', 'comp.graphics', 'sci.space', 'talk.politics.misc']

print("Loading dataset...")
data = fetch_20newsgroups(subset='all', categories=CATEGORIES,
                          remove=('headers', 'footers', 'quotes'))

# Split: 80% of posts to LEARN from, 20% hidden away to TEST on.
# It's like a student studying 80% of past exam papers, then being tested on the unseen 20%.
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target)

# The pipeline = 2 steps glued together:
# 1. TF-IDF: turns words into numbers. A word that appears a lot in THIS post
#    but rarely in others gets a high number (it "identifies" the post).
# 2. Logistic Regression: learns which number-patterns belong to which category —
#    think of it drawing a boundary line between groups.
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
])

print("Training...")
pipeline.fit(X_train, y_train)          # THE model learns here

# Now test on the unseen 20%
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {acc:.3f}")           # % of correct predictions
print(f"F1 (weighted): {f1:.3f}")       # balances precision & recall — interviewers prefer this
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Save the trained model to disk so the API can load it later
joblib.dump(pipeline, 'model.joblib')
with open('metrics.json', 'w') as f:
    json.dump({"accuracy": round(acc, 3), "f1": round(f1, 3)}, f)
print("Saved model.joblib and metrics.json")