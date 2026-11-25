# Paper recommender demo

Pick papers you like, or paste a title and abstract, and get similar papers from CiteULike-a.

## Run

```shell
./demo/run.sh
```

Needs Python 3.11+. The first run creates `demo/.venv` and installs the requirements (PyTorch
is about 2 GB on Linux). Then open http://localhost:8501.

On Windows:

```shell
cd demo
pip install -r requirements.txt
streamlit run app.py
```

## Models

| Model | What it does |
| --- | --- |
| SPECTER2 (nograph-bce) | SPECTER2 vectors with a projection trained on user saves |
| TF-IDF | Word overlap |

The score is the cosine similarity between a paper and the average of your input. Scores from
the two models are not comparable.

Free text with SPECTER2 downloads the encoder from Hugging Face the first time (about 450 MB).
Picking papers, or using TF-IDF, works offline.
