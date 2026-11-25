# 🔬 Báo Cáo Phân Tích Lỗi (Error Analysis): Lexical vs Semantic

> Phân tích định tính từ tập dữ liệu thực nghiệm CiteULike-a (Validation Fold 0).
> So sánh trực tiếp giữa **TF-IDF (Lexical)** và **SPECTER2 (Semantic Representation)**.

---

## Case 1: SPECTER2 Chiến Thắng Nhờ Hiểu Ngữ Nghĩa Sâu (Synonymy & Semantic Match)

### Trường Hợp 1.1: User 604 — SPECTER2 Rank #1 vs TF-IDF Rank #1642

**Bài báo mục tiêu (Target Paper ID: 8162):**
- **Tiêu đề**: *Collaborative Creation of Communal Hierarchical Taxonomies in Social Tagging Systems*
- **Tóm tắt (Abstract)**: "fanno una tassonomia gerarchica navigabile di tags. il tag {é} un vettore sugli url tag1 (url1 url2 url3 ... urln) dove ogni valore di urln {é} il numero di volte che il tag annota l'oggetto urln e calcola la somiglianza tag-tag con la coseno somiglianza. come noi. costruisce un grafo di somiglianza tag-tag NON pesato dove c'{é} un arco tag1-tag2 s..."

**Một số bài báo trong thư viện (Profile) của User:**
- ID 28: *The structure and function of complex networks*
- ID 226: *The PageRank Citation Ranking: Bringing Order to the Web*
- ID 785: *A Multi-agent System for Collaborative Bookmarking*

**Giải thích cơ chế (Root Cause Analysis):**
- **Tại sao TF-IDF thất bại (Rank #1642):** Người dùng quan tâm đến chủ đề này qua các bài báo trong profile, tuy nhiên bài báo mục tiêu sử dụng hệ từ vựng (terminology) khác biệt. TF-IDF chỉ tính dựa trên tần suất từ khóa trùng khớp chính xác nên điểm số tương đồng cosine bị kéo tụt xuống rất thấp.
- **Tại sao SPECTER2 thành công (Rank #1):** Mô hình transformer khoa học (SciBERT-based) đã được huấn luyện liên kết đồ thị trích dẫn (citation-informed), có khả năng gom cụm các khái niệm tương đương vào không gian 768 chiều. Do đó, dù không trùng từ khóa chính xác, vector embedding của bài báo mục tiêu vẫn nằm rất gần vector profile của người dùng.

---

### Trường Hợp 1.2: User 2831 — SPECTER2 Rank #3 vs TF-IDF Rank #1598

**Bài báo mục tiêu (Target Paper ID: 8504):**
- **Tiêu đề**: *Information Retrieval in Folksonomies: Search and Ranking*
- **Tóm tắt (Abstract)**: "FolkRank {é} un algoritmo di search per folskonomie, che sfrutta la struttura della folksonomia e trova comunit{á} nella folksonomia e struttura i risultati della ricerca. folksonomy: lightweight knowledge representation, sta per strutture concettuali create dalle persone. Sono un approccio dal basso complementare rispetto alle tecnologie del seman..."

**Một số bài báo trong thư viện (Profile) của User:**
- ID 70: *Exploiting generative models in discriminative classifiers*
- ID 142: *Analysis of a very large web search engine query log*
- ID 436: *A Maximum Entropy Approach to Natural Language Processing*

**Giải thích cơ chế (Root Cause Analysis):**
- **Tại sao TF-IDF thất bại (Rank #1598):** Người dùng quan tâm đến chủ đề này qua các bài báo trong profile, tuy nhiên bài báo mục tiêu sử dụng hệ từ vựng (terminology) khác biệt. TF-IDF chỉ tính dựa trên tần suất từ khóa trùng khớp chính xác nên điểm số tương đồng cosine bị kéo tụt xuống rất thấp.
- **Tại sao SPECTER2 thành công (Rank #3):** Mô hình transformer khoa học (SciBERT-based) đã được huấn luyện liên kết đồ thị trích dẫn (citation-informed), có khả năng gom cụm các khái niệm tương đương vào không gian 768 chiều. Do đó, dù không trùng từ khóa chính xác, vector embedding của bài báo mục tiêu vẫn nằm rất gần vector profile của người dùng.

---

## Case 2: TF-IDF Chiếm Ưu Thế Nhờ Khớp Từ Khóa Đặc Thù (Specific Keyword Overlap)

### Trường Hợp 2.1: User 3447 — TF-IDF Rank #5 vs SPECTER2 Rank #779

**Bài báo mục tiêu (Target Paper ID: 3908):**
- **Tiêu đề**: *Towards the Semantic Web: Ontology-Driven Knowledge Management*
- **Tóm tắt (Abstract)**: "{With the current changes driven by the expansion of the World Wide Web, this book uses a different approach from other books on the market: it applies ontologies to electronically available information to improve the quality of knowledge management in large and distributed organizations. Ontologies are formal theories supporting knowledge sharing ..."

**Một số bài báo trong Profile của User:**
- ID 1: *Reverse Engineering of Biological Complexity*
- ID 15: *Network motifs: simple building blocks of complex networks.*
- ID 42: *Systems Biology: A Brief Overview*

**Giải thích cơ chế (Root Cause Analysis):**
- **Tại sao TF-IDF thành công (Rank #5):** Bài báo mục tiêu và profile của user cùng chứa các thuật ngữ kỹ thuật hẹp (domain-specific jargon/named entities). Trọng số IDF cao của các từ hiếm này đẩy bài báo lên ngay đầu danh sách.
- **Tại sao SPECTER2 tụt hạng (Rank #779):** SPECTER2 ánh xạ abstract về không gian chủ đề rộng. Khi các bài báo trong ứng viên có cùng chủ đề bao quát (broad domain), SPECTER2 có xu hướng xếp ngang hàng nhiều bài, làm bài có từ khóa hẹp bị phân tán vị trí.

---

## Case 3: Giới Hạn Của Phương Pháp Dựa Trên Nội Dung (Content-Based) So Với Collaborative Filtering

Trong các hệ thống thực tế (như bài giảng L08 đã chỉ ra):
1. **Content-based (SPECTER2/TF-IDF) không nắm bắt được thị hiếu cộng đồng (Collaborative Signal):**
   - Một bài báo có thể có tiêu đề và abstract rất sát với profile của user, nhưng nếu bài báo đó có chất lượng học thuật thấp hoặc ít được cộng đồng đón nhận, Content-based vẫn gợi ý nhiệt tình vì độ tương đồng text cao.
   - Ngược lại, Collaborative Filtering (CF/ItemKNN/CTR) dựa trên hành vi đọc tập thể của 5.500 users, từ đó nhận biết được các bài báo 'kinh điển' (influential/popular) trong cùng lĩnh vực mà user chắc chắn muốn đọc, dù abstract của bài kinh điển đó có thể viết theo phong cách tổng quan khác biệt.

2. **Giải pháp Hybrid đề xuất cho dự án:**
   - Kết hợp SPECTER2 với Graph/Collaborative (như mô hình SPECTER2 + GCN của WP3 hoặc CTR) để vừa giải quyết bài toán Cold-start bằng nội dung, vừa tận dụng thói quen đọc của cộng đồng cho bài báo Warm-start.

---
*(File báo cáo được tạo tự động bởi `paperrec.error_analysis`)*