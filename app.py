import streamlit as st
from inference import classify_email

st.title("📧 AI Spam Filter (CPU-LoRA)")
st.write("Fine-tuned Qwen-0.5B running purely on CPU")

user_input = st.text_area("Enter Email/SMS content:", height=150)

if st.button("Analyze"):
    if user_input:
        with st.spinner("Analyzing..."):
            result = classify_email(user_input)
            
        if result['label'] == 'spam':
            st.error(f"🚨 SPAM DETECTED (Confidence: {result['confidence']})")
        elif result['label'] == 'ham':
            st.success(f"✅ Safe Email (Confidence: {result['confidence']})")
        else:
            st.warning("⚠️ Unknown / Unsafe Input")
            
        st.info(f"Details: {result['explanation']}")
    else:
        st.write("Please enter some text.")