import streamlit as st

st.title("Hello Streamlit")
option = st.selectbox("Pick one", ["A", "B", "C"])
st.write(f"You selected: {option}")
