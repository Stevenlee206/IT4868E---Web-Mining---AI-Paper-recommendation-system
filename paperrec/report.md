# Hướng dẫn thêm model vào PaperRec

Repo đã có loader, 5-fold warm/cold, TF-IDF, BM25 và bộ đánh giá chung.
**Dùng lại splits trong `artifacts/splits/`; thêm phương pháp mới trong `paperrec/models/`.**


## 1. Cấu trúc cần biết

| File/thư mục | Vai trò |
| --- | --- |
| `paperrec/data.py` | Đọc CiteULike-a: danh sách paper và user-item. |
| `paperrec/splits.py` | Chia folds |
| `paperrec/models/base.py` | Interface `Recommender`, lọc và xếp hạng kết quả. |
| `paperrec/models/lexical.py` | TF-IDF, BM25; tham khảo khi thêm model. |
| `paperrec/models/lsi.py` | LSI (Truncated SVD trên TF-IDF). |
| `paperrec/models/lda.py` | LDA topic modeling + xuất phân phối $\Theta$ cho CTR. |
| `paperrec/models/specter2.py` | SPECTER2 transformer embeddings + hàm demo arXiv. |
| `paperrec/evaluation.py` | Định nghĩa metrics và đánh giá từng fold. |
| `paperrec/experiment.py` | Tạo model mới mỗi fold, tổng hợp và lưu kết quả. |
| `paperrec/baselines.py` | CLI chạy hai baseline lexical. |
| `paperrec/extract_embeddings.py`| Trích xuất và xuất file `.npy` bàn giao cho WP3 và CTR. |
| `paperrec/tune.py` | Grid search siêu tham số $K$ cho LSI và LDA. |


## 2. Dữ liệu và splits

CiteULike-a có **5.551 users, 16.980 papers, 204.986 lượt lưu**.
`dataset.papers[i]` tương ứng cột `i` của ma trận; ID bắt đầu từ 0.
Text lấy từ `paper.text` (title + abstract), đọc từ `data/raw/citeulike-a/`.

Trong `artifacts/splits/{warm,cold}/fold_0` đến `fold_4`, mỗi fold chứa:

- `train.npz`, `validation.npz`, `test.npz`: **CSR user–item**, cùng shape `(5551, 16980)`.
  Giá trị 1 là đã lưu; 0 không đồng nghĩa với không thích.
- `split.json`: metadata và danh sách paper của từng partition.
- Riêng warm có `candidates.npz`: mask nén xác định các cặp được phép đánh giá.

Mỗi fold là một bộ train/validation/test hoàn chỉnh, xấp xỉ **70/10/20**, seed 42.

| Kịch bản | Cách chia |
| --- | --- |
| Warm | Chia cặp user–paper theo từng paper; paper có dưới 5 lượt lưu giữ trong train. Candidate pool khác nhau theo user. |
| Cold | Giữ lại toàn bộ tương tác của paper validation/test khỏi train; text của các paper đó vẫn có sẵn. |


## 3. Viết model mới

Tạo `paperrec/models/your_model.py`, kế thừa `Recommender` từ `.base`:

| Phương thức | Yêu cầu |
| --- | --- |
| `_fit(self)` | Học tham số hoặc chuẩn bị đặc trưng từ dữ liệu training. |
| `score(self, profile, user_index=None)` | Trả `np.ndarray` shape `(len(self.papers),)`, điểm hữu hạn cho **mọi paper**, đúng thứ tự ID; điểm cao xếp trước. |
| `get_config(self)` | Trả dict JSON gồm `super().get_config()` và các hyperparameter để lưu trong kết quả. |

`fit(papers, split)` có sẵn sẽ chuẩn bị `self.papers` (toàn catalog), `self.train`
(tương tác training), `self.train_items` (ID paper được phép học) và gọi `_fit()`.
Dùng `self.seed` cho randomness; nếu viết constructor riêng, gọi `super().__init__(seed=seed)`.


## 4. Baseline và cách chạy

| Phương pháp | Cách dùng text và lịch sử user |
| --- | --- |
| TF-IDF | Chuẩn hóa vector paper → trung bình các paper trong profile → chuẩn hóa lại → cosine similarity. |
| BM25 | Query là tập từ không trùng trong profile; chấm điểm khớp từ có điều chỉnh tần suất/độ dài. Mặc định `k1=1.2`, `b=0.75`. |

Trong terminal đã kích hoạt `.venv`, chạy từ thư mục gốc:

```shell
python -m pip install -e ".[dev]"
python -m paperrec.baselines --scenario both --partition validation --output results/lexical-validation-new
python -m pytest -q
```

Kết quả **validation** mặc định:

| Kịch bản | Model | Recall@20 | NDCG@20 |
| --- | --- | ---: | ---: |
| Warm | TF-IDF | 0.3302 ± 0.0035 | 0.2162 ± 0.0016 |
| Warm | BM25 | 0.2352 ± 0.0022 | 0.1451 ± 0.0019 |
| Cold | TF-IDF | 0.3260 ± 0.0095 | 0.2152 ± 0.0131 |
| Cold | BM25 | 0.2240 ± 0.0125 | 0.1397 ± 0.0075 |


## 5. WP2 — Semantic Models (LSI, LDA, SPECTER2)

### 5.1 Trích xuất và bàn giao Embedding (Bước 2)

Chạy script sau để sinh các file `.npy` lưu tại `artifacts/embeddings/`:

```shell
python -m paperrec.extract_embeddings --models all --output-dir artifacts/embeddings
```

- `artifacts/embeddings/specter2_embeddings.npy`: Ma trận `(16980, 768)` bàn giao cho **WP3** (làm SPECTER2+GCN).
- `artifacts/embeddings/lda_topics.npy`: Ma trận $\Theta$ `(16980, 100)` bàn giao cho bạn làm **CTR** (Collaborative Topic Regression).
- `artifacts/embeddings/lsi_embeddings.npy`: Ma trận `(16980, 200)` cho LSI.

### 5.2 Thực nghiệm và Grid Search (Bước 3)

Chạy tìm kiếm số chiều ẩn $K$ cho LSI ($K \in \{100, 200, 300\}$) và LDA ($K \in \{50, 100, 150\}$):

```shell
python -m paperrec.tune --scenario both --partition validation --output results/tuning-validation
```

Chạy có thêm SPECTER2:
```shell
python -m paperrec.tune --scenario both --partition validation --output results/tuning-validation --include-specter2
```

### 5.3 Tích hợp Demo WP4 (Bước 4)

Module `paperrec.models.specter2` cung cấp sẵn hàm `encode_arxiv_paper`:

```python
from paperrec.models.specter2 import encode_arxiv_paper

title = "Attention Is All You Need"
abstract = "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks..."
vector = encode_arxiv_paper(title, abstract)  # Trả về numpy vector (768,)
```
