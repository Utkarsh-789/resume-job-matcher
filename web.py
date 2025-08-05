# --- Finalized streamlit_app.py for Streamlit Cloud Deployment ---

import streamlit as st
import numpy as np
import joblib
import re
import fitz  # PyMuPDF
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_page_config(page_title="Resume–Job Match Predictor", layout="wide")

# Check if model and vectorizer exist
if not os.path.exists("model.pkl") or not os.path.exists("tfidf_vectorizer.pkl"):
    st.error("Model or vectorizer not found. Please ensure 'model.pkl' and 'tfidf_vectorizer.pkl' are in the same folder.")
    st.stop()

# Load model and vectorizer
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

st.title("📄 Resume–Job Match Predictor")

st.markdown("""
### How to Use:
1. Upload a **PDF Resume**.
2. Enter the **Job Description**.
3. Enter **Required Skills** (comma-separated).
4. Click **Predict Match** to see the match probability and analysis.
""")

use_example = st.checkbox("Use Example Data")

uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

job_text = st.text_area("Paste Job Description Here")
job_required_skills = st.text_input("Required Skills (comma-separated)")

if use_example:
    job_text = "We are seeking a data analyst with experience in Python, SQL, and Power BI. Must know statistics and reporting tools."
    job_required_skills = "Python, SQL, Power BI, statistics, reporting"

def extract_skills(text):
    return set(re.findall(r'\b[a-zA-Z]+\b', str(text).lower()))

def compute_features(resume, job, required_skills):
    resume_tfidf = tfidf.transform([resume])
    job_tfidf = tfidf.transform([job])
    resume_skills = extract_skills(resume)
    job_skills = {s.strip() for s in required_skills.lower().split(",") if s.strip()} if required_skills else extract_skills(job)
    skill_overlap = len(resume_skills.intersection(job_skills)) / (len(job_skills) + 1)
    features = np.hstack([resume_tfidf.toarray(), job_tfidf.toarray(), np.array([[skill_overlap]])])
    return features, resume_skills, job_skills

if st.button("🔍 Predict Match"):
    if (uploaded_file or use_example) and job_text.strip():
        try:
            resume_text = extract_text_from_pdf(uploaded_file) if uploaded_file else "Experienced data analyst with expertise in Python, SQL, Power BI, and statistics."
            features, resume_skills, job_skills = compute_features(resume_text, job_text, job_required_skills)
            probability = model.predict_proba(features)[:, 1][0]
            st.success(f"🎯 Predicted match probability: {probability:.2%}")

            # --- Visualizations ---
            st.subheader("📊 Visual Analysis")

            # Skill Overlap Pie Chart
            matched_skills = resume_skills.intersection(job_skills)
            unmatched_skills = job_skills - resume_skills
            labels = ['Matched Skills', 'Unmatched Skills']
            sizes = [len(matched_skills), len(unmatched_skills)]
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax1.axis('equal')
            st.pyplot(fig1)

            # WordCloud for Resume
            st.subheader("📄 Resume Word Cloud")
            resume_wc = WordCloud(width=800, height=400).generate(resume_text)
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.imshow(resume_wc, interpolation='bilinear')
            ax2.axis('off')
            st.pyplot(fig2)

            # WordCloud for Job Description
            st.subheader("🧾 Job Description Word Cloud")
            job_wc = WordCloud(width=800, height=400).generate(job_text)
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            ax3.imshow(job_wc, interpolation='bilinear')
            ax3.axis('off')
            st.pyplot(fig3)

        except Exception as e:
            st.error(f"❌ Error: {e}")
    else:
        st.warning("⚠️ Please upload a resume and enter a job description.")