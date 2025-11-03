import streamlit as st
from transformers import pipeline

# Load a zero-shot classification pipeline (this is the AI brain)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Title for your app
st.title("💼 MatchMaker.AI — Resume vs Job Match Tool")

# Get user input
resume = st.text_area("📝 Paste your resume here:")
job_desc = st.text_area("💼 Paste the job description here:")

# When user clicks the button
if st.button("🔍 Match Me"):
    if resume and job_desc:
        with st.spinner("🤖 Analyzing with AI..."):
            result = classifier(resume, candidate_labels=[job_desc])
            score = result["scores"][0] * 100
            st.success(f"🎯 Match Score: {round(score, 2)}%")
    else:
        st.warning("Please paste both your resume and the job description.")
