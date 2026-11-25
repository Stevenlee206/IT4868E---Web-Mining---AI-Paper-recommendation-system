# PaperRec: hướng dẫn kỹ thuật và kết quả phương pháp đồ thị (WP3)

Repo đã có loader, 5-fold warm/cold, TF-IDF, BM25 và bộ đánh giá chung.
**Dùng lại splits trong `artifacts/splits/`; thêm phương pháp mới trong `paperrec/models/`.**


## 1. Cấu trúc cần biết

| File/thư mục | Vai trò |
| --- | --- |
| `paperrec/data.py` | Đọc CiteULike-a: danh sách paper và user-item. |
| `paperrec/splits.py` | Chia folds |
| `paperrec/graph.py` | Dựng đồ thị citation và user–paper theo từng split, kèm kiểm tra leakage. |
| `paperrec/specter.py` | Tính và cache embedding SPECTER2. |
| `paperrec/models/base.py` | Interface `Recommender`, lọc và xếp hạng kết quả. |
| `paperrec/models/lexical.py` | TF-IDF, BM25; tham khảo khi thêm model. |
| `paperrec/models/lsi.py` | LSI (Truncated SVD trên TF-IDF). |
| `paperrec/models/lda.py` | LDA topic modeling. |
| `paperrec/models/specter2.py` | SPECTER2 (base) và hàm demo arXiv. |
| `paperrec/models/node2vec.py` | Node2Vec trên citation hoặc đồ thị user–paper. |
| `paperrec/models/gcn.py` | SPECTER2 thuần, Simplified PR-SGCN và bản đối chứng không đồ thị. |
| `paperrec/evaluation.py` | Định nghĩa metrics và đánh giá từng fold. |
| `paperrec/experiment.py` | Tạo model mới mỗi fold, tổng hợp và lưu kết quả. |
| `paperrec/baselines.py` | CLI chạy hai baseline chưa grid search. |
| `paperrec/run_gcn.py` | CLI chạy SPECTER2, Simplified PR-SGCN và các ablation. |
| `paperrec/analyze_degree.py` | Phân tích kết quả đã lưu theo số citation của paper. |
| `paperrec/tune_node2vec.py` | Dò p, q của Node2Vec trên validation của một fold. |

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


## 4. Đưa model vào pipeline

Sau khi hoàn thiện `YourRecommender`, tạo script chạy từ thư mục gốc:

```python
from paperrec.data import load_citeulike_a
from paperrec.splits import load_folds
from paperrec.experiment import run_experiment
from paperrec.models.your_model import YourRecommender

dataset = load_citeulike_a("data/raw/citeulike-a")
folds = load_folds("artifacts/splits/warm", dataset)  # Đổi warm thành cold khi cần.

run_experiment(
    dataset, folds,
    models={"your-model": lambda: YourRecommender(seed=42)},
    output="results/your-model-warm-validation",
    partition="validation",
)
```


## 5. Baseline và cách chạy

| Phương pháp | Cách dùng text và lịch sử user |
| --- | --- |
| TF-IDF | Chuẩn hóa vector paper → trung bình các paper trong profile → chuẩn hóa lại → cosine similarity. |
| BM25 | Query là tập từ không trùng trong profile; chấm điểm khớp từ có điều chỉnh tần suất/độ dài. Mặc định `k1=1.2`, `b=0.75`. |


Cache lexical lưu ở `artifacts/features/`: warm dùng chung features giữa folds,
cold tách theo `fold_i`. Mỗi cache có ma trận **paper–term**, vocabulary, IDF/thống kê
và metadata/checksum; không lưu profile user. Cache và kết quả sinh ra được bỏ qua bởi Git.

Trong terminal đã kích hoạt `.venv`, chạy từ thư mục gốc:

```shell
python -m pip install -e ".[dev]"
python -m paperrec.baselines --scenario both --partition validation --output results/lexical-validation-new
python -m paperrec.baselines --scenario both --partition test --output results/lexical-test-new
python -m pytest -q
python -m ruff check paperrec tests
```

Chạy validation trước, chốt cấu hình rồi mới chạy test. CLI hỗ trợ `--k1`, `--b`,
`--sublinear-tf`, `--min-df`, `--max-df`; thêm `--no-cache` để đo thời gian fit từ đầu.
Luôn chọn output mới, chương trình không ghi đè kết quả cũ. Test của phần lexical
(`tests/test_lexical.py`, WP1) hiện chưa có trong repo; `python -m pytest -q` chỉ chạy các test
đồ thị, Node2Vec và GCN bằng dữ liệu nhỏ, không phải đánh giá test split.

Kết quả **validation** mặc định ngày 05/11/2026 (mean ± sample std của 5 folds):

| Kịch bản | Model | Recall@20 | NDCG@20 |
| --- | --- | ---: | ---: |
| Warm | TF-IDF | 0.3302 ± 0.0035 | 0.2162 ± 0.0016 |
| Warm | BM25 | 0.2352 ± 0.0022 | 0.1451 ± 0.0019 |
| Cold | TF-IDF | 0.3260 ± 0.0095 | 0.2152 ± 0.0131 |
| Cold | BM25 | 0.2240 ± 0.0125 | 0.1397 ± 0.0075 |

Bảng này thuộc WP1. File kết quả không có trong repo vì Git bỏ qua `results/`, nên WP3 chưa đối chiếu
lại các số này; tái lập bằng lệnh `--partition validation` ở trên.


## 6. Đồ thị cho nhóm graph-based (WP3)

`dataset.citations` là ma trận citation vô hướng: **44.661 cạnh** sau khi bỏ 96 self-loop;
5.299 paper (31,2%) không có cạnh nào. Mỗi model lấy đồ thị từ `paperrec/graph.py`:

| Đồ thị | Warm | Cold |
| --- | --- | --- |
| `citation_training_graph` | Toàn bộ citation | Chỉ cạnh train–train (20.327–22.224 cạnh tùy fold) |
| `citation_inference_graph` | Toàn bộ citation | Thêm chiều held-out nhận từ train; không có chiều ngược lại |
| `bipartite_graph` | Lượt lưu trong `split.train` | Không dùng để xếp hạng: paper held-out không có cạnh |

Quy tắc cold: **không dùng cạnh giữa hai paper held-out**. Cờ `include_heldout_edges=True`
chỉ dành cho ablation, thêm cạnh trong cùng partition và không bao giờ nối validation với test.
Khoảng 35% paper held-out không nhận được thông điệp nào (34,7–35,6% tùy fold).

Hạn chế và giả định cần nêu khi báo cáo:

- **Citation không có hướng và không có năm xuất bản.** Cạnh của paper held-out có thể đến từ
  paper trích dẫn nó về sau, điều mà paper mới thật sự chưa có. Do đó điểm cold dùng citation
  là lạc quan, và không dùng lại các cạnh này cho demo arXiv.
- **Không dùng `item-tag.dat`.** Tag do chính user tạo khi lưu paper, nên paper chưa ai lưu
  thường không có tag.
- **Cache embedding phải có `graph_fingerprint`** vì `dataset_fingerprint` không bao gồm citation.

**Simplified PR-SGCN.** SPECTER2+GCN trong dự án này là phiên bản đơn giản hóa, điều chỉnh cho
CiteULike-a, **không phải bản tái lập đầy đủ** PR-SGCN của Guo et al. (2025). Theo mô tả bài báo
mà nhóm đã tra cứu, phương pháp gốc dùng mạng thông tin không đồng nhất (citation, semantic,
paper–author), temporal knowledge graph và margin-based loss. CiteULike-a không có dữ liệu
paper–author và timestamp, nên phiên bản này chỉ giữ feature SPECTER2, GCN trên citation và
margin-based loss; xem kết quả ở mục 8.

Kiểm tra đồ thị trên các fold đã lưu và chạy test:

```shell
python -m paperrec.graph --scenario both
python -m pytest tests/test_graph.py -q
```


## 7. Node2Vec

`paperrec/models/node2vec.py`: random walk bậc hai (p, q) viết bằng NumPy, train skip-gram bằng
gensim. Profile user là trung bình các embedding paper đã chuẩn hóa; điểm là cosine.
Paper cold nhận trung bình embedding các hàng xóm thuộc train. Paper không có hàng xóm đã có
embedding được xếp sau mọi paper có embedding, theo thứ tự tie ngẫu nhiên cố định bằng seed.

```shell
python -m pip install -e ".[graph]"
python -m paperrec.run_node2vec --output results/node2vec-validation-new
python -m pytest tests/test_node2vec.py -q
```

Kết quả **validation** ngày 06/11/2026, mean ± sample std của 5 folds, cấu hình mặc định **chưa
tinh chỉnh** (p=q=1, 128 chiều, 10 walk, độ dài 80, window 10, 1 epoch, seed 42):

| Kịch bản | Đồ thị | Recall@20 | NDCG@20 | Train mỗi lần |
| --- | --- | ---: | ---: | ---: |
| Warm | Citation | 0.3151 ± 0.0026 | 0.2020 ± 0.0008 | ~56 giây |
| Warm | Bipartite | 0.4711 ± 0.0036 | 0.3198 ± 0.0020 | ~120 giây |
| Cold | Citation | 0.3182 ± 0.0079 | 0.2242 ± 0.0064 | ~34 giây |

- Warm citation dùng chung một embedding cho cả 5 fold, nên độ lệch chuẩn không phản ánh độ
  ngẫu nhiên khi train; thời gian trung bình trong `results.json` cũng bị cache làm thấp đi.
  Thêm `--no-cache` để đo thời gian train thật.
- Warm citation: 5.299 paper (31,2%) không có embedding. Cold: 36,3–37,5% paper held-out không có
  embedding, cao hơn tỷ lệ không có cạnh đến train vì một số hàng xóm train lại cô lập trong đồ
  thị train.
- `catalog_coverage` chia cho toàn catalog, trong khi pool cold chỉ khoảng 10% catalog.
- Gensim 4.4 có thể in `Exception ignored in: ...our_dot_float` không kèm traceback. Đây là cảnh
  báo giả khi tích vô hướng đúng bằng -1,0, không phải lỗi train.


## 8. SPECTER2 và Simplified PR-SGCN

`paperrec/specter.py` tính embedding 768 chiều bằng `allenai/specter2_base` với adapter
`allenai/specter2` (proximity), batch 8. Cache nằm ở `artifacts/features/specter2/`, ghi dần
xuống đĩa nên chạy tiếp được nếu bị ngắt. Toàn bộ 16.980 paper mất khoảng 13 phút trên Apple MPS.

`paperrec/models/gcn.py` có ba loại model:

| Tên trong kết quả | Cách xếp hạng |
| --- | --- |
| `specter2` | Cosine trên SPECTER2 gốc, không train. |
| `gcn-margin`, `gcn-bce` | Simplified PR-SGCN: GCN trên citation, train với margin loss hoặc binary cross-entropy. |
| `nograph-margin`, `nograph-bce` | Cùng mạng và loss nhưng chỉ có self-loop, để đo riêng phần đóng góp của citation. |

**Vì sao dùng PyTorch thuần.** Đồ thị inference của cold là một chiều: paper held-out nhận thông
điệp từ train, không gửi đi. Chuẩn hóa GCN thông thường giả định ma trận đối xứng và sẽ làm đổi
trọng số train–train khi thêm paper held-out. Dự án dùng trọng số riêng
`w(i←j) = 1 / sqrt(d_i d_j)`, với `d_i = 1 + số thông điệp i nhận`, trong
`graph.normalized_message_matrix`. Paper train nhận cùng số thông điệp ở hai đồ thị, nên trọng số
của chúng giữ nguyên giữa train và inference. Viết layer quanh ma trận tường minh này bằng
PyTorch thuần giúp dễ đọc, dễ kiểm tra và dễ trình bày quy tắc; `tests/test_gcn.py` kiểm tra rằng
cạnh của paper held-out không làm đổi embedding của paper train.

**Huấn luyện.** Mỗi user lấy một paper đã lưu làm positive, các paper đã lưu còn lại làm profile
(tối đa 50), cùng 5 negative lấy từ `training_mask()` trừ các paper user đã lưu trong train. Vì vậy
cặp held-out của warm và paper held-out của cold không bao giờ bị xem là negative. Không dùng early
stopping.

```shell
python -m pip install -e ".[graph,neural]"
python -m paperrec.specter
python -m paperrec.run_gcn --output results/gcn-validation-new
python -m pytest tests/test_gcn.py -q
```

Kết quả **validation** ngày 06/11/2026, mean ± sample std của 5 folds, seed 42. Cấu hình: 1 layer,
output 128 chiều, 50 epoch, lr 1e-3, dropout 0,1, margin 0,2, scale BCE 10.

| Kịch bản | Model | Recall@20 | NDCG@20 | Train mỗi fold |
| --- | --- | ---: | ---: | ---: |
| Warm | SPECTER2 | 0.4371 ± 0.0038 | 0.2793 ± 0.0025 | 0 |
| Warm | GCN margin | 0.4893 ± 0.0061 | 0.3224 ± 0.0038 | ~73 giây |
| Warm | GCN BCE | 0.5059 ± 0.0051 | 0.3375 ± 0.0022 | ~73 giây |
| Warm | Không đồ thị, margin | 0.4981 ± 0.0069 | 0.3242 ± 0.0031 | ~61 giây |
| Warm | Không đồ thị, BCE | **0.5241 ± 0.0053** | **0.3534 ± 0.0028** | ~64 giây |
| Cold | SPECTER2 | 0.4223 ± 0.0094 | 0.2692 ± 0.0112 | 0 |
| Cold | GCN margin | 0.4741 ± 0.0088 | 0.3119 ± 0.0083 | ~66 giây |
| Cold | GCN BCE | 0.4973 ± 0.0085 | 0.3316 ± 0.0084 | ~66 giây |
| Cold | Không đồ thị, margin | 0.4865 ± 0.0077 | 0.3131 ± 0.0076 | ~63 giây |
| Cold | Không đồ thị, BCE | **0.5160 ± 0.0081** | **0.3462 ± 0.0086** | ~61 giây |

Chênh lệch theo từng cặp fold (GCN trừ bản không đồ thị):

| Kịch bản | Loss | Recall@20 | Số fold GCN thấp hơn | NDCG@20 | Số fold GCN thấp hơn |
| --- | --- | ---: | ---: | ---: | ---: |
| Warm | BCE | -0.0182 | 5/5 | -0.0159 | 5/5 |
| Warm | Margin | -0.0088 | 5/5 | -0.0019 | 3/5 |
| Cold | BCE | -0.0188 | 5/5 | -0.0145 | 5/5 |
| Cold | Margin | -0.0124 | 5/5 | -0.0012 | 3/5 |

Kết luận trên validation:

- **Citation không cải thiện kết quả ở kiến trúc GCN thuần này.** Với BCE, GCN thấp hơn bản không đồ
  thị ở mọi fold. Với margin, Recall@20 cũng thấp hơn ở mọi fold; NDCG@20 không có khác biệt rõ.
  Mục 9 cho thấy thêm residual đảo ngược kết quả này.
- **Phần tăng so với SPECTER2 đến từ phép chiếu học trên lượt lưu của user.** Bản không đồ thị 1 layer
  chỉ là một phép chiếu tuyến tính, nhưng đã tăng Recall@20 thêm 0.087 (warm) và 0.094 (cold) so
  với SPECTER2 gốc.
- **BCE tốt hơn margin ở mọi fold**, cả khi có và không có đồ thị.
- SPECTER2 gốc đã cao hơn Node2Vec citation: Recall@20 hơn 0.122 (warm) và 0.104 (cold).

Hạn chế khi đọc các số này:

- Cấu hình được chọn trên validation của fold 0, và fold này cũng nằm trong kết quả, nên điểm có phần
  lạc quan. Bước chọn thử bản có đồ thị nhiều hơn, nên không nghiêng về bản không đồ thị.
- Margin và BCE không được dò đều nhau. Tất cả chỉ chạy một seed, nên độ lệch chuẩn phản ánh
  khác biệt giữa các fold, không phản ánh độ ngẫu nhiên khi train.
- Citation cold có thể chứa cạnh từ tương lai (mục 6), nên điểm GCN cold có thể lạc quan.
- SPECTER2 được huấn luyện sẵn trên nguồn ngoài, có thể đã thấy một phần paper trong CiteULike-a.
- Kết quả test nằm ở mục 10.


## 9. Ablation và phân tích theo citation (validation)

Mọi ablation chỉ đổi một yếu tố so với `gcn-bce` (1 layer, 50 epoch, seed 42), trên cùng 5 fold:

| Tên | Thay đổi |
| --- | --- |
| `gcn-bce-add` | Đầu vào của layer là `H + A H` thay vì `A H`. |
| `gcn-bce-concat` | Đầu vào của layer là `[H, A H]`; có thể biểu diễn bản không đồ thị. |
| `gcn-bce-drop25/50/75` | Bỏ ngẫu nhiên 25/50/75% cạnh citation khỏi đồ thị, cả khi train và inference. Đây không phải DropEdge lấy mẫu lại mỗi epoch. |
| `gcn-bce-knn5/20` | Thay citation bằng k paper train gần nhất theo cosine SPECTER2. |

Đồ thị kNN chỉ dùng text. Hàng xóm luôn là paper train, nên cạnh train–train không phụ thuộc text của
paper held-out; quy tắc một chiều của mục 6 vẫn áp dụng. Với k=5, chỉ khoảng 16% cạnh citation trùng
với cạnh kNN.

```shell
python -m paperrec.run_gcn --output results/ablation-validation/residual --models gcn-bce-add gcn-bce-concat
python -m paperrec.run_gcn --output results/ablation-validation/edgedrop --models gcn-bce-drop25 gcn-bce-drop50 gcn-bce-drop75
python -m paperrec.run_gcn --output results/ablation-validation/knn5 --models gcn-bce-knn5
python -m paperrec.run_gcn --output results/ablation-validation/knn20 --models gcn-bce-knn20
python -m paperrec.analyze_degree --results results/gcn-validation results/ablation-validation/residual --models nograph-bce gcn-bce gcn-bce-add gcn-bce-concat --output results/degree-validation/residual-new.json
python -m paperrec.tune_node2vec --output results/node2vec-tuning-new.json
```

Kết quả **validation** ngày 06–07/11/2026, mean ± sample std của 5 folds. Chênh lệch là model trừ
`nograph-bce` theo từng cặp fold, kèm số fold model cao hơn:

| Kịch bản | Model | Recall@20 | NDCG@20 | Chênh R@20 | Fold cao hơn | Chênh NDCG@20 | Fold cao hơn |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Warm | Không đồ thị BCE | 0.5241 ± 0.0053 | 0.3534 ± 0.0028 | | | | |
| Warm | GCN BCE | 0.5059 ± 0.0051 | 0.3375 ± 0.0022 | -0.0182 | 0/5 | -0.0159 | 0/5 |
| Warm | **Residual add** | **0.5354 ± 0.0058** | **0.3676 ± 0.0019** | +0.0112 | 5/5 | +0.0142 | 5/5 |
| Warm | Residual concat | 0.5326 ± 0.0057 | 0.3634 ± 0.0023 | +0.0085 | 5/5 | +0.0100 | 5/5 |
| Warm | Bỏ 25% cạnh | 0.5049 ± 0.0051 | 0.3365 ± 0.0021 | -0.0193 | 0/5 | -0.0169 | 0/5 |
| Warm | Bỏ 50% cạnh | 0.5038 ± 0.0056 | 0.3345 ± 0.0024 | -0.0203 | 0/5 | -0.0189 | 0/5 |
| Warm | Bỏ 75% cạnh | 0.5049 ± 0.0038 | 0.3317 ± 0.0023 | -0.0192 | 0/5 | -0.0217 | 0/5 |
| Warm | kNN k=5 | 0.5242 ± 0.0045 | 0.3544 ± 0.0017 | +0.0001 | 3/5 | +0.0010 | 4/5 |
| Warm | kNN k=20 | 0.5128 ± 0.0054 | 0.3423 ± 0.0035 | -0.0113 | 0/5 | -0.0111 | 0/5 |
| Cold | Không đồ thị BCE | 0.5160 ± 0.0081 | 0.3462 ± 0.0086 | | | | |
| Cold | GCN BCE | 0.4973 ± 0.0085 | 0.3316 ± 0.0084 | -0.0188 | 0/5 | -0.0145 | 0/5 |
| Cold | **Residual add** | **0.5297 ± 0.0103** | **0.3617 ± 0.0086** | +0.0137 | 5/5 | +0.0155 | 5/5 |
| Cold | Residual concat | 0.5237 ± 0.0090 | 0.3547 ± 0.0079 | +0.0077 | 5/5 | +0.0085 | 5/5 |
| Cold | Bỏ 25% cạnh | 0.4945 ± 0.0099 | 0.3271 ± 0.0082 | -0.0215 | 0/5 | -0.0191 | 0/5 |
| Cold | Bỏ 50% cạnh | 0.4925 ± 0.0090 | 0.3251 ± 0.0078 | -0.0235 | 0/5 | -0.0211 | 0/5 |
| Cold | Bỏ 75% cạnh | 0.4939 ± 0.0092 | 0.3239 ± 0.0074 | -0.0221 | 0/5 | -0.0223 | 0/5 |
| Cold | kNN k=5 | 0.5104 ± 0.0081 | 0.3407 ± 0.0056 | -0.0056 | 0/5 | -0.0055 | 1/5 |
| Cold | kNN k=20 | 0.4954 ± 0.0081 | 0.3287 ± 0.0038 | -0.0206 | 0/5 | -0.0174 | 0/5 |

**Theo số citation của paper held-out.** Hit rate@20 gộp trên các cặp user–paper held-out, trung bình
5 fold; số citation lấy từ đồ thị inference nghiêm ngặt. Vì gộp theo cặp, số này khác Recall@20 macro.

| Kịch bản | Model | 0 | 1 | 2–3 | 4–7 | 8+ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Warm | Không đồ thị BCE | 0.4651 | 0.4118 | 0.4352 | 0.4611 | 0.4603 |
| Warm | GCN BCE | 0.3637 | 0.3363 | 0.3908 | 0.4803 | 0.5410 |
| Warm | Residual add | 0.4027 | 0.3678 | 0.4260 | 0.4935 | 0.5530 |
| Warm | kNN k=5 | 0.4715 | 0.4225 | 0.4433 | 0.4599 | 0.4523 |
| Cold | Không đồ thị BCE | 0.4586 | 0.4141 | 0.4343 | 0.4625 | 0.4511 |
| Cold | GCN BCE | 0.3658 | 0.3506 | 0.4179 | 0.5165 | 0.5165 |
| Cold | Residual add | 0.3948 | 0.3817 | 0.4418 | 0.5200 | 0.5601 |
| Cold | kNN k=5 | 0.4608 | 0.4181 | 0.4410 | 0.4638 | 0.4206 |

Tỷ lệ cặp held-out theo các nhóm trên: warm 30/8/14/18/31%, cold 32/10/17/19/22% (làm tròn).

**Node2Vec, dò p và q.** Lưới `{0.25, 1, 4}²` trên validation fold 0, sau đó chạy 5 fold. Fold 1–4
không tham gia bước chọn nên là phần so sánh không thiên lệch với mặc định p=q=1:

| Kịch bản | Đồ thị | p, q chọn | Recall@20 | NDCG@20 | Chênh R@20 fold 1–4 | Chênh NDCG@20 fold 1–4 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Warm | Citation | 4, 1 | 0.3187 ± 0.0036 | 0.2038 ± 0.0010 | +0.0030 | +0.0017 |
| Warm | Bipartite | 0.25, 1 | 0.4765 ± 0.0037 | 0.3240 ± 0.0021 | +0.0057 | +0.0043 |
| Cold | Citation | 1, 0.25 | 0.3174 ± 0.0072 | 0.2235 ± 0.0066 | -0.0010 | -0.0009 |

Một số ô của warm bipartite trùng điểm hoàn toàn, ví dụ (0.25, 1) và (1, 4). Đây không phải lỗi:
đồ thị hai phía không có tam giác nên chỉ tỷ lệ q/p quyết định bước đi. Ô (1, 1) và (4, 4) có cùng phân
phối nhưng khác đường lấy mẫu, và lệch nhau 0.0005 Recall@20 trên fold 0. Warm citation dùng chung
một embedding cho mọi fold, nên việc cả 5 fold cùng tăng không phải 5 bằng chứng độc lập.

Kết luận trên validation:

- **Giữ đặc trưng riêng của paper làm citation có ích.** Residual add cao hơn bản không đồ thị ở mọi fold.
- **Lợi ích không phân bố đều.** So với bản không đồ thị, residual add tăng mạnh ở paper nhiều citation
  nhưng vẫn giảm 0.062 (warm) và 0.064 (cold) hit rate ở paper không có citation.
- **Paper mới thật sự thuộc nhóm 0 citation.** Điểm cold của residual add đến từ nhóm có citation, và các
  cạnh đó có thể từ tương lai (mục 6). Trong dữ liệu này, ở nhóm paper không có citation, bản không đồ
  thị tốt hơn residual add. Xem giới hạn cold-start ở mục 10.
- **Bỏ cạnh không đưa điểm dần về bản không đồ thị.** Bỏ 75% cạnh vẫn cho Recall@20 gần như GCN đầy đủ,
  trong khi bỏ 100% tương đương bản không đồ thị (test xác nhận trên đồ thị nhỏ). Số lượng cạnh vì thế
  không giải thích được phần điểm mất.
- **kNN k=5 không vượt bản không đồ thị:** hòa ở warm, thấp hơn ở cold. Nó vẫn cao hơn GCN citation thuần,
  và mọi paper đều có hàng xóm. Giả thuyết chưa kiểm chứng: GCN thuần bất lợi cho paper cô lập vì đối xử
  khác nhau giữa paper có và không có cạnh, thay vì do bản thân citation. Chưa thử residual với kNN.
- **p, q gần như không đáng kể.** Mức tăng lớn nhất là 0.0057 Recall@20, và cold còn giảm nhẹ. Khi báo cáo
  cold nên giữ p=q=1. Node2Vec vẫn thấp hơn rõ so với các model dùng SPECTER2.

Hạn chế khi đọc các số này:

- Các ablation chạy một seed. Riêng residual add và bản không đồ thị có thêm seed 43, 44 ở phần dưới.
  Residual add là model tốt nhất trong nhiều lựa chọn trên cùng validation, nên điểm validation của nó
  có phần lạc quan; xem kết quả test ở mục 10.
- Phân tích theo citation là quan sát: số citation có thể đi cùng độ phổ biến hoặc chủ đề, và top-20 của
  mỗi user có giới hạn nên tăng ở nhóm này kéo theo giảm ở nhóm khác.
- Lượt chạy bị gián đoạn do máy ngủ và có tiến trình chạy song song, nên thời gian train của lượt này
  không dùng để so sánh. Kết quả không đổi khi chạy lại: fold 0 của `gcn-bce` cho đúng điểm đã lưu.
- Chưa chạy: DropEdge lấy mẫu lại mỗi epoch và residual với kNN.

**Độ ổn định theo seed (validation, 5 folds mỗi seed).** Residual add trừ bản không đồ thị:

| Kịch bản | Metric | Seed 42 | Seed 43 | Seed 44 | Fold cao hơn ở mỗi seed |
| --- | --- | ---: | ---: | ---: | ---: |
| Warm | Recall@20 | +0.0112 | +0.0125 | +0.0119 | 5/5 |
| Warm | NDCG@20 | +0.0142 | +0.0151 | +0.0138 | 5/5 |
| Cold | Recall@20 | +0.0137 | +0.0131 | +0.0136 | 5/5 |
| Cold | NDCG@20 | +0.0155 | +0.0151 | +0.0150 | 5/5 |

**Cạnh giữa hai paper held-out (cold, validation).** So với đồ thị nghiêm ngặt, thêm cạnh trong cùng
partition làm Recall@20 đổi -0.0006 với GCN BCE và +0.0005 với residual add; đều trong khoảng dao động
giữa các fold. Xem kết quả test ở mục 10.

### Hạn chế của mô hình

**Residual add vẫn bất lợi cho paper không có citation, và chuẩn hóa `A·H` không khắc phục được.** Đã thử
hai cách trên residual add ở validation fold 0: chia theo bậc (`input_norm="row"`, trọng số `1 / d_i`) và
chuẩn hóa L2 từng hàng của `A·H` (`input_norm="l2"`). Tiêu chí được chốt trước khi chạy: hit rate nhóm 0
citation phải tăng ở cả warm và cold thì mới chạy 5 fold.

| Kịch bản | Biến thể | Recall@20 | NDCG@20 | Hit rate nhóm 0 | Chênh so với residual add |
| --- | --- | ---: | ---: | ---: | ---: |
| Warm | Residual add | 0.5375 | 0.3701 | 0.4083 | |
| Warm | L2 | 0.5356 | 0.3691 | 0.3952 | -0.0131 |
| Warm | Chia theo bậc | 0.5354 | 0.3679 | 0.3899 | -0.0184 |
| Cold | Residual add | 0.5353 | 0.3696 | 0.4098 | |
| Cold | L2 | 0.5351 | 0.3694 | 0.4039 | -0.0059 |
| Cold | Chia theo bậc | 0.5326 | 0.3666 | 0.3938 | -0.0160 |

Cả hai đều không đạt nên không chạy 5 fold và không đưa vào model. Kết quả này cho thấy độ lớn của đầu vào
không phải nguyên nhân chính. Vì sao paper không có citation bị bất lợi vẫn chưa được xác định. Hai tùy chọn
vẫn ở trong code để tái lập phép thử. Số của hai biến thể không được lưu trong `results/`; lệnh dưới đây tạo file kết
quả trên validation (5 fold, trong đó có fold 0):

```shell
python -m paperrec.run_gcn --output results/ablation-validation/input-norm --models gcn-bce-add-l2 gcn-bce-add-row
```


## 10. Kết quả trên test

Danh sách model được khóa trước khi chạy test, và mỗi cấu hình chỉ đánh giá test một lần, không có bước
chọn hay sửa model nào sau đó. Mọi model train trên `split.train`, seed 42, không gộp validation vào train.

**`gcn-bce-add` được phát triển dựa trên quan sát từ validation**, sau khi phân tích mục 9 cho thấy GCN thuần
thấp hơn bản không đồ thị. Nó là phần bổ sung của dự án, không thuộc mô tả PR-SGCN gốc mà nhóm đã tra cứu.
`gcn-bce` mới là đại diện Simplified PR-SGCN; `nograph-bce` là đối chứng.

**SPECTER2 thuần được bổ sung sau**, khi đã có kết quả test của các model khác, để làm mốc so sánh cho feature
đầu vào của GCN. Model này không train và không có tham số để chọn, nên việc thêm nó không đổi cấu hình nào.

Mean ± sample std của 5 folds (test, 07/11/2026):

| Kịch bản | Model | Recall@20 | NDCG@20 | Recall@10 | NDCG@10 | MRR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Warm | **Residual add** | **0.4150 ± 0.0018** | **0.3213 ± 0.0014** | 0.2959 | 0.2853 | 0.4326 |
| Warm | Không đồ thị BCE | 0.4019 ± 0.0021 | 0.3052 ± 0.0017 | 0.2833 | 0.2687 | 0.4075 |
| Warm | Simplified PR-SGCN (GCN BCE) | 0.3820 ± 0.0026 | 0.2899 ± 0.0009 | 0.2656 | 0.2538 | 0.3893 |
| Warm | Node2Vec bipartite (p=0.25, q=1) | 0.3584 ± 0.0032 | 0.2801 ± 0.0016 | 0.2497 | 0.2480 | 0.3765 |
| Warm | SPECTER2 thuần | 0.3223 ± 0.0020 | 0.2338 ± 0.0017 | 0.2212 | 0.2006 | 0.3174 |
| Warm | Node2Vec citation (p=4, q=1) | 0.2355 ± 0.0021 | 0.1692 ± 0.0003 | 0.1607 | 0.1433 | 0.2352 |
| Cold | **Residual add** | **0.4000 ± 0.0069** | **0.3123 ± 0.0041** | 0.2858 | 0.2778 | 0.4276 |
| Cold | Không đồ thị BCE | 0.3873 ± 0.0036 | 0.2973 ± 0.0017 | 0.2744 | 0.2627 | 0.4037 |
| Cold | Simplified PR-SGCN (GCN BCE) | 0.3676 ± 0.0088 | 0.2811 ± 0.0062 | 0.2551 | 0.2460 | 0.3853 |
| Cold | SPECTER2 thuần | 0.3120 ± 0.0028 | 0.2289 ± 0.0033 | 0.2152 | 0.1969 | 0.3155 |
| Cold | Node2Vec citation (p=q=1) | 0.2463 ± 0.0078 | 0.1972 ± 0.0048 | 0.1779 | 0.1776 | 0.3088 |

Cold dùng p=q=1 vì cấu hình dò được trên fold 0 thấp hơn mặc định khi chạy 5 fold validation.

Chênh lệch theo từng cặp fold so với `nograph-bce`:

| Kịch bản | Model | Chênh R@20 | Fold cao hơn | Chênh NDCG@20 | Fold cao hơn |
| --- | --- | ---: | ---: | ---: | ---: |
| Warm | Residual add | +0.0131 | 5/5 | +0.0161 | 5/5 |
| Warm | GCN BCE | -0.0199 | 0/5 | -0.0152 | 0/5 |
| Cold | Residual add | +0.0127 | 5/5 | +0.0150 | 5/5 |
| Cold | GCN BCE | -0.0198 | 0/5 | -0.0162 | 0/5 |

Chênh lệch theo từng cặp fold so với SPECTER2 thuần:

| Kịch bản | Model | Chênh R@20 | Fold cao hơn | Chênh NDCG@20 | Fold cao hơn |
| --- | --- | ---: | ---: | ---: | ---: |
| Warm | Residual add | +0.0927 | 5/5 | +0.0875 | 5/5 |
| Warm | Không đồ thị BCE | +0.0796 | 5/5 | +0.0714 | 5/5 |
| Warm | GCN BCE | +0.0597 | 5/5 | +0.0561 | 5/5 |
| Warm | Node2Vec bipartite | +0.0361 | 5/5 | +0.0463 | 5/5 |
| Warm | Node2Vec citation | -0.0868 | 0/5 | -0.0646 | 0/5 |
| Cold | Residual add | +0.0880 | 5/5 | +0.0835 | 5/5 |
| Cold | Không đồ thị BCE | +0.0753 | 5/5 | +0.0684 | 5/5 |
| Cold | GCN BCE | +0.0555 | 5/5 | +0.0522 | 5/5 |
| Cold | Node2Vec citation | -0.0658 | 0/5 | -0.0317 | 0/5 |

Hit rate@20 theo số citation trên test (mô tả sau khi đã có kết quả, không dùng để chọn model):

| Kịch bản | Model | 0 | 1 | 2–3 | 4–7 | 8+ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Warm | SPECTER2 thuần | 0.2517 | 0.1950 | 0.2082 | 0.2355 | 0.2749 |
| Warm | Không đồ thị BCE | 0.3284 | 0.2868 | 0.3046 | 0.3362 | 0.3306 |
| Warm | GCN BCE | 0.2205 | 0.2043 | 0.2634 | 0.3584 | 0.4207 |
| Warm | Residual add | 0.2596 | 0.2389 | 0.2929 | 0.3678 | 0.4353 |
| Cold | SPECTER2 thuần | 0.2345 | 0.2037 | 0.2159 | 0.2384 | 0.2784 |
| Cold | Không đồ thị BCE | 0.3115 | 0.2976 | 0.3078 | 0.3256 | 0.3195 |
| Cold | GCN BCE | 0.2187 | 0.2335 | 0.2924 | 0.3889 | 0.3885 |
| Cold | Residual add | 0.2459 | 0.2615 | 0.3117 | 0.3866 | 0.4360 |

**Ablation cạnh held-out trên test (cold).** So với đồ thị nghiêm ngặt, thêm cạnh giữa các paper test:

| Model | Recall@20 | Chênh R@20 | Fold cao hơn | NDCG@20 | Chênh NDCG@20 | Fold cao hơn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GCN BCE | 0.3648 ± 0.0079 | -0.0028 | 0/5 | 0.2779 ± 0.0061 | -0.0032 | 0/5 |
| Residual add | 0.4002 ± 0.0073 | +0.0002 | 3/5 | 0.3123 ± 0.0045 | 0.0000 | 2/5 |

Quy tắc nghiêm ngặt không làm giảm điểm: thêm các cạnh này không giúp ích cho cả hai model.

Kết luận:

- **Kết quả validation lặp lại trên test.** Residual add cao hơn bản không đồ thị ở mọi fold, với mức chênh
  gần như trên validation. GCN thuần vẫn thấp hơn bản không đồ thị ở mọi fold.
- **Citation chỉ có ích khi giữ đặc trưng riêng của paper,** và phần lợi tập trung ở paper có nhiều citation.
- **Phần tăng lớn nhất so với SPECTER2 thuần đến từ phép chiếu học trên lượt lưu của user:** bản không đồ thị
  tăng Recall@20 thêm 0.0796 (warm) và 0.0753 (cold). Residual add chỉ thêm khoảng 0.013 trên mức đó.
- **Node2Vec citation thấp hơn SPECTER2 thuần ở mọi fold,** còn Node2Vec bipartite cao hơn ở warm; nó dùng lượt
  lưu của user nên không áp dụng được cho paper cold.

### Giới hạn ở kịch bản cold-start

Điểm cold ở trên là của paper chưa có lượt lưu nhưng **vẫn có thể có citation**, không phải của paper mới thật sự.
Vì vậy không kết luận rằng `gcn-bce-add` giải quyết được cold-start.

- **Điểm cold của residual add đến từ paper đã có citation.** Khoảng 34% cặp held-out thuộc nhóm 0 citation. Ở nhóm
  này và nhóm 1 citation, bản không đồ thị cao hơn residual add ở 5/5 fold (nhóm 0: 0.3115 so với 0.2459);
  ở nhóm 4–7 và 8+, residual add cao hơn ở 5/5 fold (nhóm 8+: 0.4360 so với 0.3195).
- **Paper mới thật sự không có citation.** Citation không có hướng và năm xuất bản (mục 6), nên cạnh của paper
  held-out có thể đến từ paper trích dẫn nó về sau.
- **Ở nhóm 0 citation, phần lợi của residual add so với SPECTER2 thuần rất nhỏ:** cold +0.0114 (4/5 fold), warm
  +0.0079 (5/5 fold). GCN BCE còn thấp hơn SPECTER2 thuần ở nhóm này: cold -0.0158 (1/5 fold cao hơn), warm -0.0312
  (0/5 fold). Bản không đồ thị cao hơn SPECTER2 thuần ở 5/5 fold, khoảng 0.077 ở cả hai kịch bản.
- **Khuyến nghị cho demo arXiv: dùng `nograph-bce` cho paper chưa có citation.** Đây là suy luận từ nhóm 0 citation
  của CiteULike-a, chưa được kiểm chứng trên arXiv. Paper 0 citation trong dữ liệu này có thể khác paper mới về
  chủ đề hoặc độ phổ biến, nên đây chỉ là xấp xỉ.

### Hạn chế chung và lệnh tái lập

Hạn chế khi đọc các số này:

- **Không so Recall@20 giữa test và validation.** Test giữ khoảng 20% tương tác, gấp đôi validation, nên mỗi
  user có nhiều positive hơn trong khi top-20 có giới hạn. Recall vì thế thấp hơn, còn MRR không giảm; chênh
  lệch theo từng cặp fold mới là phép so sánh phù hợp.
- Trên test chỉ chạy seed 42; độ ổn định theo seed mới được kiểm tra trên validation.
- 5 fold dùng chung dữ liệu nên không độc lập; số fold cao hơn không phải kiểm định thống kê.
- Citation của cold có thể chứa cạnh từ tương lai (mục 6), nên điểm cold của các model dùng citation có thể
  lạc quan. Các hạn chế về SPECTER2 ở mục 8 và hạn chế của mô hình ở mục 9 vẫn áp dụng.
- TF-IDF và BM25 thuộc WP1, chưa được đánh giá trên test trong phần này.

Các lệnh đã dùng; **không chạy lại để chọn model**, vì test chỉ dành cho đánh giá cuối:

```shell
python -m paperrec.run_gcn --partition test --output results/test/gcn --models gcn-bce nograph-bce gcn-bce-add
python -m paperrec.run_gcn --partition test --output results/test/heldout --scenario cold --models gcn-bce-heldout gcn-bce-add-heldout
python -m paperrec.run_node2vec --partition test --output results/test/node2vec-warm-citation --scenario warm --graph citation --p 4 --q 1
python -m paperrec.run_node2vec --partition test --output results/test/node2vec-warm-bipartite --scenario warm --graph bipartite --p 0.25 --q 1
python -m paperrec.run_node2vec --partition test --output results/test/node2vec-cold-citation --scenario cold --graph citation --p 1 --q 1
python -m paperrec.analyze_degree --results results/test/gcn --models nograph-bce gcn-bce gcn-bce-add --output results/test/degree-new.json
python -m paperrec.run_gcn --partition test --output results/test/specter2 --models specter2
python -m paperrec.analyze_degree --results results/test/specter2 results/test/gcn --models specter2 nograph-bce gcn-bce-add --output results/test/degree-specter2.json
```


## 11. WP2 — Semantic Models (LSI, LDA, SPECTER2)

### 11.1 Trích xuất và bàn giao Embedding (Bước 2)

Chạy script sau để sinh các file `.npy` lưu tại `artifacts/embeddings/`:

```shell
python -m paperrec.extract_embeddings --models all --output-dir artifacts/embeddings
```

- `artifacts/embeddings/specter2_embeddings.npy`: Ma trận `(16980, 768)` bàn giao cho **WP3** (làm SPECTER2+GCN).
- `artifacts/embeddings/lda_topics.npy`: Ma trận $\Theta$ `(16980, 100)` bàn giao cho bạn làm **CTR** (Collaborative Topic Regression).
- `artifacts/embeddings/lsi_embeddings.npy`: Ma trận `(16980, 200)` cho LSI.

### 11.2 Thực nghiệm và Grid Search (Bước 3)

Chạy tìm kiếm số chiều ẩn $K$ cho LSI ($K \in \{100, 200, 300\}$) và LDA ($K \in \{50, 100, 150\}$):

```shell
python -m paperrec.tune --scenario both --partition validation --output results/tuning-validation
```

Chạy có thêm SPECTER2:
```shell
python -m paperrec.tune --scenario both --partition validation --output results/tuning-validation --include-specter2
```

### 11.3 Tích hợp Demo WP4 (Bước 4)

Module `paperrec.models.specter2` cung cấp sẵn hàm `encode_arxiv_paper`:

```python
from paperrec.models.specter2 import encode_arxiv_paper

title = "Attention Is All You Need"
abstract = "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks..."
vector = encode_arxiv_paper(title, abstract)  # Trả về numpy vector (768,)
```
