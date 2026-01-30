import streamlit as st
from model import predict_single, predict_batch

# Page config
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
        prediction = predict_single(review_text)
        st.success(f"Prediction: **{prediction}**")

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
        predictions = predict_batch(reviews)

        for r, p in zip(reviews, predictions):
            st.write(f"**Review:** {r}")
            st.write(f"➡ Prediction: `{p}`")
            st.markdown("---")

# Footer
st.markdown("💡 *Predictions are automatically saved to the database/file.*")

