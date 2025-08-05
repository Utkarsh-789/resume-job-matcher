# --- Enhanced preprocessing.py with EDA, Cleaning & Visualizations ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

# Load data
resumes = pd.read_csv('resume.csv')
jobs = pd.read_csv('jobs.csv')

# --- Data Cleaning ---
resumes['Resume_str'] = resumes['Resume_str'].astype(str).str.lower().str.strip()
jobs['Job Description'] = jobs['Job Description'].astype(str).str.lower().str.strip()
jobs['Required Skills'] = jobs['Required Skills'].astype(str).str.lower().str.strip()

# Remove duplicates
resumes.drop_duplicates(subset='Resume_str', inplace=True)
jobs.drop_duplicates(subset='Job Description', inplace=True)

# --- Skill Extraction ---
def extract_skills(text):
    return set(re.findall(r'\b[a-zA-Z]+\b', str(text).lower()))

resumes['Skills_set'] = resumes['Resume_str'].apply(extract_skills)
jobs['Skills_set'] = jobs['Required Skills'].apply(extract_skills)

# --- Exploratory Data Analysis ---
print("\nResume Categories:")
if 'Category' in resumes.columns:
    print(resumes['Category'].value_counts())
    resumes['Category'].value_counts().plot(kind='bar', title='Resume Categories', figsize=(10, 4))
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

print("\nTop Job Titles:")
if 'Job Title' in jobs.columns:
    print(jobs['Job Title'].value_counts().head(10))
    jobs['Job Title'].value_counts().head(10).plot(kind='bar', title='Top 10 Job Titles', figsize=(10, 4))
    plt.xlabel('Job Title')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

# WordCloud: Resume Text
all_resume_text = ' '.join(resumes['Resume_str'])
wordcloud = WordCloud(width=800, height=400).generate(all_resume_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Resume Word Cloud')
plt.show()

# WordCloud: Job Descriptions
all_job_text = ' '.join(jobs['Job Description'])
wordcloud = WordCloud(width=800, height=400).generate(all_job_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Job Description Word Cloud')
plt.show()

# --- Resume–Job Pairing & Labeling ---
pairs = []
labels = []
for i, resume in resumes.iterrows():
    for j, job in jobs.iterrows():
        skill_overlap = len(resume['Skills_set'].intersection(job['Skills_set']))
        pairs.append({
            'resume_text': resume['Resume_str'],
            'job_text': job['Job Description'],
            'skill_overlap': skill_overlap / (len(job['Skills_set']) + 1)
        })
        labels.append(int(skill_overlap >= 4))

pairs_df = pd.DataFrame(pairs)
pairs_df['label'] = labels

# Skill Overlap Distribution
plt.hist(pairs_df['skill_overlap'], bins=30, color='skyblue')
plt.title('Skill Overlap Distribution')
plt.xlabel('Skill Overlap')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# --- TF-IDF Vectorization ---
all_text = pairs_df['resume_text'].tolist() + pairs_df['job_text'].tolist()
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf.fit(all_text)
resume_tfidf = tfidf.transform(pairs_df['resume_text'])
job_tfidf = tfidf.transform(pairs_df['job_text'])

features = hstack([resume_tfidf, job_tfidf, np.array(pairs_df['skill_overlap']).reshape(-1, 1)])

# --- Model Training ---
X_train, X_test, y_train, y_test = train_test_split(features, pairs_df['label'], test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

# --- Evaluation ---
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print("\nModel Performance")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))

# --- Save Model & Vectorizer ---
joblib.dump(model, 'model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')