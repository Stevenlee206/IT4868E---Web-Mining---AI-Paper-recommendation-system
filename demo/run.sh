#!/bin/sh
cd "$(dirname "$0")" || exit 1
[ -d .venv ] || python3 -m venv .venv || exit 1
.venv/bin/pip install -q -r requirements.txt || exit 1
exec .venv/bin/streamlit run app.py
