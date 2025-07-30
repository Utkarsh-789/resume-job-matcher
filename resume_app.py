# resume_app.py
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import re

# Load saved label names
labels = pd.read_csv("label_classes.csv", header=None).squeeze().tolist()

# Clean resume input
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower()

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("bert_resume_model")
    model = BertForSequenceClassification.from_pretrained("bert_resume_model")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit UI
st.set_page_config(page_title="Resume Classifier", layout="wide")
st.title("🤖 Resume to Job Role Classifier")
st.write("Paste your resume below and get the most likely job category prediction.")

resume_input = st.text_area("✍️ Paste Resume Text Here", height=300)

if st.button("Predict"):
    if not resume_input.strip():
        st.warning("Please paste some resume content.")
    else:
        cleaned = clean_text(resume_input)
        inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        st.success(f"🎯 Predicted Job Category: **{labels[prediction]}**")
