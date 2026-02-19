import streamlit as st
from model import predict_single, predict_batch

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🕵️",
    layout="centered"
)
