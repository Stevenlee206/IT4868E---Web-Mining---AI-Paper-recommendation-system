import hf
import streamlit as st
from models import HOSTED, MODELS, TEXT_MODELS, papers, recommend

st.set_page_config(page_title="Paper recommender", page_icon="📄")
st.title("Paper recommender")

with st.sidebar:
    label = st.selectbox("Model", list(MODELS))
    k = st.slider("Top K", 1, 50, 10)

name = MODELS[label]
takes_text = label in TEXT_MODELS
catalog = papers()

if not takes_text:
    st.info(f"{label} only knows papers in the list, so text input is off.")
picked = st.multiselect("Papers you like", range(len(catalog)), format_func=lambda i: catalog[i][0])
title = st.text_input("Title").strip() if takes_text else ""
abstract = st.text_area("Abstract").strip() if takes_text else ""
if name in HOSTED:
    st.caption("Typed text is sent to a Hugging Face Space, which may need a minute to wake up.")

if st.button("Recommend", type="primary"):
    if not (picked or title or abstract):
        st.warning("Pick a paper" + (" or enter some text." if takes_text else "."))
        st.stop()
    try:
        results = recommend(name, k, picked, title, abstract)
    except hf.Unavailable as error:
        st.warning(f"The encoder is unavailable ({error}). Showing TF-IDF results instead.")
        results = recommend("tfidf", k, picked, title, abstract)
    if not results:
        st.warning("This model has nothing to match on for this input.")
    for rank, (i, score) in enumerate(results, 1):
        left, right = st.columns([5, 1])
        left.markdown(f"**{rank}. {catalog[i][0]}**")
        right.markdown(f"`{score:.3f}`")
        with left.expander("Abstract"):
            st.write(catalog[i][1])
