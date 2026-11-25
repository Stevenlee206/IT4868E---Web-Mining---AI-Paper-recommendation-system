"""Compare the Space with the stored vectors of one paper (uses one run)."""

import hf
from models import papers, unit, vectors

try:
    result = hf.vectors(*papers()[0])
except hf.Unavailable as error:
    raise SystemExit(f"Unavailable: {error}") from None
for key in ("base", "adapter"):
    cosine = unit(result[key]) @ vectors(f"specter2-{key}")[0]
    print(f"{key}: cosine {cosine:.4f} ({'ok' if cosine > 0.98 else 'MISMATCH'})")
