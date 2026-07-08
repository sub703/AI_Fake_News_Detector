import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import joblib
import re
import numpy as np

# Set page title and layout configuration
st.set_page_config(page_title="AI Fake News Detector", layout="centered")

# Define the exact text-cleaning function used during training
DATELINE_RE  = re.compile(r"^[A-Z][\w .,'\-]{0,40}?\(reuters\)\s*[-\u2013\u2014]*", re.IGNORECASE)
WIRE_RE      = re.compile(r"\(reuters\)|\(ap\)", re.IGNORECASE)
URL_RE       = re.compile(r"http\S+|www\.\S+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
SPACE_RE     = re.compile(r"\s+")

def clean_text(text: str) -> str:
    text = str(text)
    text = DATELINE_RE.sub(" ", text)   # Remove "CITY (Reuters) -" openers
    text = text[:4000].lower()          # Cap length and lowercase
    text = WIRE_RE.sub(" ", text)       # Remove residual wire tags
    text = URL_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)  # Keep letters and spaces only
    text = SPACE_RE.sub(" ", text).strip()
    return text

# Load the trained pipeline artifact (cached so it only loads once)
@st.cache_resource
def load_pipeline():
    # Make sure 'fake_news_pipeline.joblib' is in the same directory or adjust path
    return joblib.load("fake_news_pipeline.joblib")

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Could not load the model pipeline file: {e}")
    st.info("Please ensure 'fake_news_pipeline.joblib' is placed in the same folder as this script.")
    st.stop()

# Application User Interface Header
st.title("🛡️ AI Fake News Detector")
st.write("Enter a news headline and article body below to evaluate its linguistic authenticity using our calibrated Linear SVM pipeline.")

# Input fields matching the original deployment behavior
title_input = st.text_input("Article Title / Headline:", placeholder="e.g., Global oil prices stabilize as production limits extend")
body_input = st.text_area("Article Body Text:", placeholder="e.g., The energy committee announced on Wednesday that...", height=150)

CONFIDENCE_FLOOR = 0.57  # Tuned threshold from section 13 for 97% target precision

if st.button("Analyze News Authenticity", type="primary"):
    combined = f"{title_input}. {body_input}".strip()
    cleaned = clean_text(combined)
    
    if not cleaned:
        st.warning("⚠️ Please provide text in the headline or body fields to generate a prediction.")
    else:
        # Extract classification probabilities
        proba = pipeline.predict_proba([cleaned])[0]
        classes = pipeline.classes_
        
        # Determine top class index
        fake_idx = list(classes).index("fake")
        fake_probability = float(proba[fake_idx])
        
        # Apply the adjusted decision threshold logic
        if fake_probability >= CONFIDENCE_FLOOR:
            st.error(f"🚨 **Likely FAKE** ({fake_probability:.1%} confidence score)")
            st.markdown(
                "**Linguistic Indicators:** This text contains structural patterns, metadata calls, or sensational phrases commonly observed in fabricated media segments."
            )
        elif (1.0 - fake_probability) >= CONFIDENCE_FLOOR:
            real_probability = 1.0 - fake_probability
            st.success(f"✅ **Likely REAL** ({real_probability:.1%} confidence score)")
            st.markdown(
                "**Linguistic Indicators:** This text mirrors standard objective journalistic frameworks, editorial calendar references, and formal structures typical of authentic reports."
            )
        else:
            # Fallback if neither class clears the strict confidence target floor
            max_prob = max(fake_probability, 1.0 - fake_probability)
            st.info(f"❓ **Uncertain, verify independently** ({max_prob:.1%} marginal confidence)")
            st.write("The text structural signals are too subtle or mixed to guarantee a highly precise classification layout.")

st.markdown("---")
st.caption("Disclaimer: This model evaluates stylistic and structural vocabulary patterns; it does not explicitly verify historical external facts.")
