import streamlit as st
from recommender import Specter, Tfidf, load_papers, recommend

st.set_page_config(page_title="Paper recommender", page_icon="📄")

MODELS = {"TF-IDF": lambda: Tfidf(papers()), "SPECTER2 (nograph-bce)": Specter}


@st.cache_resource(show_spinner="Loading papers...")
def papers():
    return load_papers()


@st.cache_resource(show_spinner="Loading model...")
def get_model(name):
    return MODELS[name]()


st.title("Paper recommender")
st.caption("16,980 papers from CiteULike-a")

with st.sidebar:
    name = st.selectbox("Model", list(MODELS), index=1)
    k = st.slider("Top K", 1, 50, 10)

catalog = papers()
selected = st.multiselect(
    "Papers you like", range(len(catalog)), format_func=lambda i: catalog[i].title
)
with st.expander("Or describe a paper"):
    title = st.text_input("Title")
    abstract = st.text_area("Abstract")

if st.button("Recommend", type="primary"):
    if not (selected or title.strip() or abstract.strip()):
        st.warning("Pick at least one paper or enter some text.")
        st.stop()
    with st.spinner("Ranking..."):
        results = recommend(get_model(name), k, selected, title.strip(), abstract.strip())
    for rank, (i, score) in enumerate(results, 1):
        left, right = st.columns([5, 1])
        left.markdown(f"**{rank}. {catalog[i].title}**")
        right.markdown(f"`{score:.3f}`")
        with left.expander("Abstract"):
            st.write(catalog[i].abstract)
