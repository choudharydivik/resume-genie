import streamlit as st
from PIL import Image

logo = Image.open("ResumeGenieLogo.png")
st.sidebar.image(logo,width=140)

st.sidebar.markdown("**Hello World**")

st.title("Hello World")
st.write("Welcome to my first streamlit App")