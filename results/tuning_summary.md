# 📊 Báo Cáo Kết Quả Thực Nghiệm WP2: Semantic Methods (LSI, LDA, SPECTER2)

> Đánh giá 5-fold Cross-Validation trên tập CiteULike-a (5.551 users, 16.980 papers, 204.986 saves).
> So sánh các mô hình ngữ nghĩa (WP2) với các baseline từ khóa Lexical (WP1).

---

## 1. Kịch Bản Warm-Start (Có Lịch Sử Tương Tác)

| Model | Config | Recall@5 | Recall@10 | Recall@20 | NDCG@20 | MRR | Train Time |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **TF-IDF (Baseline WP1)** | L2 norm | 0.1712 | 0.2453 | 0.3302 ± 0.0035 | 0.2162 ± 0.0016 | 0.2601 | ~1s |
| **BM25 (Baseline WP1)** | k1=1.2, b=0.75 | 0.1104 | 0.1642 | 0.2352 ± 0.0022 | 0.1451 ± 0.0019 | 0.1796 | ~2s |
| lsi_k100 | - | 0.1523 | 0.2315 | 0.3286 ± 0.0014 | 0.2066 ± 0.0017 | 0.2489 | 6.0s |
| lsi_k200 | - | 0.1671 | 0.2460 | 0.3381 ± 0.0039 | 0.2191 ± 0.0021 | 0.2639 | 7.4s |
| **lsi_k300** | - | 0.1742 | 0.2513 | 0.3416 ± 0.0020 | 0.2239 ± 0.0020 | 0.2704 | 9.7s |
| lda_k50 | - | 0.0762 | 0.1257 | 0.2003 ± 0.0042 | 0.1171 ± 0.0027 | 0.1511 | 143.4s |
| lda_k100 | - | 0.0730 | 0.1191 | 0.1867 ± 0.0027 | 0.1104 ± 0.0014 | 0.1461 | 258.1s |
| lda_k150 | - | 0.0777 | 0.1272 | 0.2003 ± 0.0031 | 0.1181 ± 0.0012 | 0.1518 | 305.6s |
| **specter2** | - | 0.2096 | 0.3094 | 0.4316 ± 0.0034 | 0.2744 ± 0.0027 | 0.3090 | 0.1s |

---

## 2. Kịch Bản Cold-Start (Bài Báo Mới Hoàn Toàn)

| Model | Config | Recall@5 | Recall@10 | Recall@20 | NDCG@20 | MRR | Train Time |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **TF-IDF (Baseline WP1)** | L2 norm | 0.1680 | 0.2419 | 0.3260 ± 0.0095 | 0.2152 ± 0.0131 | 0.2629 | ~1s |
| **BM25 (Baseline WP1)** | k1=1.2, b=0.75 | 0.1068 | 0.1578 | 0.2240 ± 0.0125 | 0.1397 ± 0.0075 | 0.1732 | ~2s |
| lsi_k100 | - | 0.1464 | 0.2283 | 0.3260 ± 0.0094 | 0.2027 ± 0.0052 | 0.2412 | 3.3s |
| lsi_k200 | - | 0.1614 | 0.2445 | 0.3354 ± 0.0100 | 0.2163 ± 0.0053 | 0.2617 | 4.8s |
| **lsi_k300** | - | 0.1686 | 0.2499 | 0.3409 ± 0.0112 | 0.2217 ± 0.0054 | 0.2664 | 6.0s |
| lda_k50 | - | 0.0742 | 0.1247 | 0.1993 ± 0.0132 | 0.1154 ± 0.0080 | 0.1486 | 74.7s |
| lda_k100 | - | 0.0761 | 0.1240 | 0.1958 ± 0.0094 | 0.1161 ± 0.0060 | 0.1539 | 142.0s |
| lda_k150 | - | 0.0762 | 0.1264 | 0.1998 ± 0.0071 | 0.1178 ± 0.0045 | 0.1531 | 200.9s |
| **specter2** | - | 0.2028 | 0.2991 | 0.4198 ± 0.0074 | 0.2655 ± 0.0066 | 0.2985 | 0.1s |

---

## 3. Phân Tích & Kết Luận Chính

1. **SPECTER2 (allenai/specter2_base)** là mô hình vượt trội nhất:
   - **Warm-start**: Đạt Recall@20 = **0.4316** (vượt TF-IDF +10.14% tuyệt đối, tương đương mức tăng trưởng **+30.7%**).
   - **Cold-start**: Đạt Recall@20 = **0.4198** (vượt TF-IDF +9.38% tuyệt đối, tương đương mức tăng trưởng **+28.8%**).
   - Nhờ pre-computed embeddings (`specter2_embeddings.npy`), thời gian huấn luyện và suy luận chỉ mất **0.1s/fold**, đáp ứng hoàn hảo tiêu chí < 3s/user.

2. **LSI (Truncated SVD)** cải thiện ổn định theo số chiều ẩn $K$:
   - Cả hai kịch bản đều cho thấy xu hướng rõ rệt: $K=300 > K=200 > K=100$.
   - Với $K=300$, LSI đạt Recall@20 = **0.3416** (Warm) và **0.3409** (Cold), đều vượt trội so với TF-IDF gốc (0.3302 và 0.3260).
   - Việc giảm chiều ma trận giúp khử nhiễu từ vựng đồng nghĩa và từ hiếm hiệu quả.

3. **LDA (Latent Dirichlet Allocation)**:
   - Điểm số dừng ở mức ~0.20 do việc nén 17.000 chiều từ vựng xuống 50-150 chủ đề xác suất làm mờ đi ranh giới phân biệt giữa các bài báo cụ thể.
   - Tuy nhiên, ma trận phân phối chủ đề $\Theta$ trích xuất được (`lda_topics.npy`) là thành phần thiết yếu được chuyển giao thành công cho mô hình Collaborative Topic Regression (CTR).

---
*(5-fold cross-validation, phần validation. LSI, LDA và SPECTER2 chạy bằng `paperrec.tune`; Recall@5, Recall@10 và MRR của TF-IDF, BM25 đã được sửa theo lần chạy lại bằng `paperrec.benchmark`.)*