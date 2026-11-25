# Paper recommender demo

Recommends CiteULike-a papers with 13 models. Everything is pre-computed, so there is no PyTorch
and a query takes a few milliseconds on CPU.

## Run

```shell
cd demo
pip install -r requirements.txt && streamlit run app.py
```

Or `./demo/run.sh`, which also creates a virtual environment. Tested on Python 3.12.

## Typed text

TF-IDF, BM25, LSI and LDA work offline. SPECTER2 and the GCN models send the text to a Hugging
Face Space that runs the encoder (`space/`). Its address is in `space.txt`, or in `SPACE_URL`.

The Space runs on free ZeroGPU hardware. It sleeps when unused, and calls without a token share
2 GPU minutes a day. Set `HF_TOKEN` to a read token to get your own quota. If the Space is asleep,
out of quota or unreachable, the app shows TF-IDF results and says so.

To host your own (free accounts need a verified email and to be older than 30 days):

```shell
pip install huggingface_hub
read -rs "HF_TOKEN?Write token: " && export HF_TOKEN
python deploy_space.py
unset HF_TOKEN
```

ItemKNN, WMF and Node2Vec only know papers in the list, so their text box is hidden.

## Good to know

- All models were trained on warm fold 0 of the benchmark.
- A typed paper has no citations, so the GCN models only see its own text.
- WMF estimates a vector for you from the papers you pick.
- Scores are cosines, except for BM25, ItemKNN and WMF. Do not compare scores between models.

`demo_data/` is built on Kaggle with `python scripts/kaggle/launch.py demo`.
