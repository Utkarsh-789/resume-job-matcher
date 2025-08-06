import streamlit as st
import numpy as np
import joblib
import re
import PyPDF2
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Load model and vectorizer ---
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

st.title("Resume–Job Match Predictor")

# Upload multiple resumes
uploaded_files = st.file_uploader("Upload Resume PDFs (multiple allowed)", type="pdf", accept_multiple_files=True)

# Input job description and skills
job_text = st.text_area("Paste Job Description Here")
job_required_skills = st.text_input("Required Skills (comma-separated, as in job)")

# -----------------------------
# Helper functions
# -----------------------------

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_skills(text):
    return set(re.findall(r'\b[a-zA-Z]+\b', str(text).lower()))

def compute_features(resume, job, required_skills):
    resume_tfidf = tfidf.transform([resume])
    job_tfidf = tfidf.transform([job])
    resume_skills = extract_skills(resume)
    job_skills = {s.strip() for s in required_skills.lower().split(",") if s.strip()} if required_skills else extract_skills(job)
    skill_overlap = len(resume_skills.intersection(job_skills)) / (len(job_skills) + 1)
    features = np.hstack([resume_tfidf.toarray(), job_tfidf.toarray(), np.array([[skill_overlap]])])
    return features, skill_overlap

# -----------------------------
# Prediction and Display
# -----------------------------
if st.button("Predict Matches"):
    if not job_text.strip():
        st.warning("Please enter a job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume PDF.")
    else:
        results = []

        for uploaded_file in uploaded_files:
            resume_text = extract_text_from_pdf(uploaded_file)
            features, overlap = compute_features(resume_text, job_text, job_required_skills)
            probability = model.predict_proba(features)[:,1][0]
            results.append({
                "filename": uploaded_file.name,
                "probability": probability,
                "overlap": overlap,
                "text": resume_text
            })

        # Sort by match probability in descending order
        results = sorted(results, key=lambda x: x["probability"], reverse=True)

        # Display results
        for res in results:
            st.subheader(f"📄 {res['filename']}")
            st.write(f"✅ **Match Probability:** {res['probability']:.2%}")
            st.write(f"🧠 **Skill Overlap Score:** {res['overlap']:.2f}")
            st.markdown("---")

        # Show summary chart
        st.subheader("📊 Match Probability Comparison (Sorted)")
        fig, ax = plt.subplots()
        filenames = [res["filename"] for res in results]
        probabilities = [res["probability"] for res in results]
        ax.barh(filenames[::-1], probabilities[::-1])  # Reverse to show top match at top
        ax.set_xlabel("Match Probability")
        ax.set_xlim(0, 1)
        st.pyplot(fig)



