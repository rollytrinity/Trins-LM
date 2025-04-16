import streamlit as st

st.title("✅ Hello Streamlit!")
st.write("If you see this, the app is running properly!")

if st.button("Click me"):
    st.success("Button works!")
