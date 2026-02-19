import streamlit as st
from model import predict_single, predict_batch

# ---------------- PAGE CONFIG ----------------0
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🕵️",
    layout="centered"
)

st.title("🕵️ Fake Review Detection System")
st.write("Detect whether a product review is **Fake** or **Genuine** using NLP.")

# ---------------- SINGLE PREDICTION ----------------
st.subheader("🔍 Single Review Prediction")

review_text = st.text_area("Enter a review:")

if st.button("Predict Review"):
    if review_text.strip() == "":
        st.warning("Please enter a review.")
    else:
        label, confidence = predict_single(review_text)

        if label == "FAKE_REVIEW":
            st.error(f"🚨 Prediction: **{label}**")
        elif label == "GENUINE_REVIEW":
            st.success(f"✅ Prediction: **{label}**")
        else:
            st.warning(f"⚠️ Prediction: **{label}**")

        st.info(f"Confidence: **{confidence}%**")

# ---------------- BATCH PREDICTION ----------------
st.subheader("📦 Batch Review Prediction")

batch_text = st.text_area(
    "Enter multiple reviews (one per line):",
    height=150
)

if st.button("Predict Batch"):
    reviews = [line.strip() for line in batch_text.split("\n") if line.strip()]

    if not reviews:
        st.warning("Please enter at least one review.")
    else:
        results = predict_batch(reviews)

        for review, (label, confidence) in zip(reviews, results):
            st.write(f"**Review:** {review}")

            if label == "FAKE_REVIEW":
                st.error(f"🚨 {label} ({confidence}%)")
            elif label == "GENUINE_REVIEW":
                st.success(f"✅ {label} ({confidence}%)")
            else:
                st.warning(f"⚠️ {label} ({confidence}%)")

            st.markdown("---")

# ---------------- FOOTER ----------------
st.markdown("💡 *Predictions are automatically saved for future analysis.*")

