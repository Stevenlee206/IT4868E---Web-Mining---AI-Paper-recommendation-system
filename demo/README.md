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

TF-IDF, BM25, LSI and LDA work offline. SPECTER2 and the GCN models encode text with the
Hugging Face API, which needs a free [token](https://huggingface.co/settings/tokens) allowed to
call Inference Providers:

```shell
export HF_TOKEN=hf_...
```

Without a token, without internet, or when the API fails, the app shows TF-IDF results and says
so. The text you type is sent to Hugging Face only for these models.

ItemKNN, WMF and Node2Vec only know papers in the list, so their text box is hidden.

## Good to know

- All models were trained on warm fold 0 of the benchmark.
- The API serves SPECTER2 Base only. Adapter vectors for typed text are predicted from base
  vectors with a linear map (0.991 mean cosine on held-out papers). Picked papers are not affected.
- A typed paper has no citations, so the GCN models only see its own text.
- WMF estimates a vector for you from the papers you pick.
- Scores are cosines, except for BM25, ItemKNN and WMF. Do not compare scores between models.

`demo_data/` is built on Kaggle with `python scripts/kaggle/launch.py demo`.
