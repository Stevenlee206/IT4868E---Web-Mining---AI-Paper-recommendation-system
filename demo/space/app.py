import gradio as gr
import spaces
import torch
from adapters import AutoAdapterModel
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
model.load_adapter("allenai/specter2", source="hf", load_as="proximity")
model = model.to(device).eval()


def cls(text):
    inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.inference_mode():
        return model(**inputs).last_hidden_state[0, 0].float().cpu().tolist()


@spaces.GPU(duration=10)
def encode(title, abstract):
    title, abstract = title.strip()[:1000], abstract.strip()[:10000]
    model.set_active_adapters(None)
    base = cls(f"{title} [SEP] {abstract}")
    model.set_active_adapters("proximity")
    return {"base": base, "adapter": cls(f"{title}{tokenizer.sep_token}{abstract}")}


gr.Interface(
    encode,
    [gr.Textbox(label="Title"), gr.Textbox(label="Abstract", lines=6)],
    gr.JSON(label="Vectors"),
    title="SPECTER2 encoder",
    api_name="encode",
    flagging_mode="never",
).launch()
