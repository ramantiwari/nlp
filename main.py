import streamlit as st
from transformers import pipeline
import PyPDF2
import matplotlib.pyplot as plt

# Load model (first time slow hoga)
emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

# PDF text extract
def extract_text(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# Emotion detection
def detect_emotion(text):
    result = emotion_model(text[:512])

    # case handle
    if isinstance(result, list):
        result = result[0]

    # agar dict hai (single output)
    if isinstance(result, dict):
        return {result['label']: result['score']}

    # agar list of dict hai
    elif isinstance(result, list):
        return {r['label']: r['score'] for r in result}

    return {}

# Document analysis
def analyze_document(text):
    sentences = text.split(".")
    final_scores = {}

    for sentence in sentences[:50]:  # limit for speed
        emotions = detect_emotion(sentence)

        for key, value in emotions.items():
            final_scores[key] = final_scores.get(key, 0) + value

    # normalize
    total = sum(final_scores.values())
    for key in final_scores:
        final_scores[key] = round((final_scores[key] / total) * 100, 2)

    return final_scores

# UI
st.title("AI PDF Emotion Analyzer")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:
    st.write("Processing...")

    text = extract_text(uploaded_file)

    if text:
        result = analyze_document(text)

        st.subheader("Emotion Analysis")
        st.write(result)

        # Graph
        fig, ax = plt.subplots()
        ax.bar(result.keys(), result.values())
        plt.xticks(rotation=45)
        st.pyplot(fig)

    else:
        st.error("Text extract nahi ho paaya ")