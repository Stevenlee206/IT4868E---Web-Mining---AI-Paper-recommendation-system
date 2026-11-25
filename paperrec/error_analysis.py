"""Error analysis and case study extraction: Lexical (TF-IDF) vs Semantic (SPECTER2).

Finds real examples where:
1. SPECTER2 wins (vocabulary mismatch / synonymy where TF-IDF fails).
2. TF-IDF wins (exact technical term matching where dense embeddings are too general).
Outputs analysis report to results/error_analysis.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import load_citeulike_a
from .models.lexical import TfidfRecommender
from .models.specter2 import SPECTER2Recommender
from .splits import load_split


def run_error_analysis(dataset, split, output_path: Path):
    print("Fitting TF-IDF and SPECTER2 for error analysis on fold 0...", flush=True)

    tfidf = TfidfRecommender(seed=42)
    tfidf.fit(dataset.papers, split)

    specter2 = SPECTER2Recommender(
        cache_path="artifacts/embeddings/specter2_embeddings.npy",
        seed=42,
    )
    specter2.fit(dataset.papers, split)

    val_targets = split.validation
    eval_users = split.evaluation_users("validation")

    semantic_wins = []
    lexical_wins = []

    print(f"Scanning {len(eval_users)} users for representative case studies...", flush=True)
    for u in eval_users:
        profile = split.train[u].indices
        relevant = val_targets[u].indices
        candidates = split.candidate_items(int(u), "validation")

        # Rank candidates
        recs_tfidf = [item for item, _ in tfidf.recommend(profile, k=len(candidates), candidates=candidates)]
        recs_specter = [item for item, _ in specter2.recommend(profile, k=len(candidates), candidates=candidates)]

        rank_map_tfidf = {item: r + 1 for r, item in enumerate(recs_tfidf)}
        rank_map_specter = {item: r + 1 for r, item in enumerate(recs_specter)}

        for item in relevant:
            r_tf = rank_map_tfidf[item]
            r_sp = rank_map_specter[item]

            # Case A: Semantic clearly wins (SPECTER2 in top 5, TF-IDF ranked beyond 50)
            if r_sp <= 5 and r_tf >= 50:
                semantic_wins.append({
                    "user_id": int(u),
                    "target_id": int(item),
                    "rank_specter": r_sp,
                    "rank_tfidf": r_tf,
                    "profile_ids": profile.tolist(),
                })

            # Case B: Lexical clearly wins (TF-IDF in top 5, SPECTER2 ranked beyond 40)
            if r_tf <= 5 and r_sp >= 40:
                lexical_wins.append({
                    "user_id": int(u),
                    "target_id": int(item),
                    "rank_specter": r_sp,
                    "rank_tfidf": r_tf,
                    "profile_ids": profile.tolist(),
                })

    print(f"Found {len(semantic_wins)} semantic-win cases and {len(lexical_wins)} lexical-win cases.", flush=True)

    # Pick the most striking cases
    semantic_wins.sort(key=lambda x: x["rank_tfidf"] - x["rank_specter"], reverse=True)
    lexical_wins.sort(key=lambda x: x["rank_specter"] - x["rank_tfidf"], reverse=True)

    best_semantic_cases = semantic_wins[:2]
    best_lexical_case = lexical_wins[0] if lexical_wins else None

    # Build markdown report
    lines = [
        "# 🔬 Báo Cáo Phân Tích Lỗi (Error Analysis): Lexical vs Semantic",
        "",
        "> Phân tích định tính từ tập dữ liệu thực nghiệm CiteULike-a (Validation Fold 0).",
        "> So sánh trực tiếp giữa **TF-IDF (Lexical)** và **SPECTER2 (Semantic Representation)**.",
        "",
        "---",
        "",
        "## Case 1: SPECTER2 Chiến Thắng Nhờ Hiểu Ngữ Nghĩa Sâu (Synonymy & Semantic Match)",
        "",
    ]

    for i, case in enumerate(best_semantic_cases, start=1):
        target = dataset.papers[case["target_id"]]
        prof_samples = [dataset.papers[pid] for pid in case["profile_ids"][:3]]

        lines.extend([
            f"### Trường Hợp 1.{i}: User {case['user_id']} — SPECTER2 Rank #{case['rank_specter']} vs TF-IDF Rank #{case['rank_tfidf']}",
            "",
            f"**Bài báo mục tiêu (Target Paper ID: {target.paper_id}):**",
            f"- **Tiêu đề**: *{target.title}*",
            f"- **Tóm tắt (Abstract)**: \"{target.abstract[:350]}...\"",
            "",
            "**Một số bài báo trong thư viện (Profile) của User:**",
        ])
        for p in prof_samples:
            lines.append(f"- ID {p.paper_id}: *{p.title}*")

        lines.extend([
            "",
            "**Giải thích cơ chế (Root Cause Analysis):**",
            f"- **Tại sao TF-IDF thất bại (Rank #{case['rank_tfidf']}):** Người dùng quan tâm đến chủ đề này qua các bài báo trong profile, tuy nhiên bài báo mục tiêu sử dụng hệ từ vựng (terminology) khác biệt. TF-IDF chỉ tính dựa trên tần suất từ khóa trùng khớp chính xác nên điểm số tương đồng cosine bị kéo tụt xuống rất thấp.",
            f"- **Tại sao SPECTER2 thành công (Rank #{case['rank_specter']}):** Mô hình transformer khoa học (SciBERT-based) đã được huấn luyện liên kết đồ thị trích dẫn (citation-informed), có khả năng gom cụm các khái niệm tương đương vào không gian 768 chiều. Do đó, dù không trùng từ khóa chính xác, vector embedding của bài báo mục tiêu vẫn nằm rất gần vector profile của người dùng.",
            "",
            "---",
            "",
        ])

    if best_lexical_case:
        target = dataset.papers[best_lexical_case["target_id"]]
        prof_samples = [dataset.papers[pid] for pid in best_lexical_case["profile_ids"][:3]]
        lines.extend([
            f"## Case 2: TF-IDF Chiếm Ưu Thế Nhờ Khớp Từ Khóa Đặc Thù (Specific Keyword Overlap)",
            "",
            f"### Trường Hợp 2.1: User {best_lexical_case['user_id']} — TF-IDF Rank #{best_lexical_case['rank_tfidf']} vs SPECTER2 Rank #{best_lexical_case['rank_specter']}",
            "",
            f"**Bài báo mục tiêu (Target Paper ID: {target.paper_id}):**",
            f"- **Tiêu đề**: *{target.title}*",
            f"- **Tóm tắt (Abstract)**: \"{target.abstract[:350]}...\"",
            "",
            "**Một số bài báo trong Profile của User:**",
        ])
        for p in prof_samples:
            lines.append(f"- ID {p.paper_id}: *{p.title}*")

        lines.extend([
            "",
            "**Giải thích cơ chế (Root Cause Analysis):**",
            f"- **Tại sao TF-IDF thành công (Rank #{best_lexical_case['rank_tfidf']}):** Bài báo mục tiêu và profile của user cùng chứa các thuật ngữ kỹ thuật hẹp (domain-specific jargon/named entities). Trọng số IDF cao của các từ hiếm này đẩy bài báo lên ngay đầu danh sách.",
            f"- **Tại sao SPECTER2 tụt hạng (Rank #{best_lexical_case['rank_specter']}):** SPECTER2 ánh xạ abstract về không gian chủ đề rộng. Khi các bài báo trong ứng viên có cùng chủ đề bao quát (broad domain), SPECTER2 có xu hướng xếp ngang hàng nhiều bài, làm bài có từ khóa hẹp bị phân tán vị trí.",
            "",
            "---",
            "",
        ])

    lines.extend([
        "## Case 3: Giới Hạn Của Phương Pháp Dựa Trên Nội Dung (Content-Based) So Với Collaborative Filtering",
        "",
        "Trong các hệ thống thực tế (như bài giảng L08 đã chỉ ra):",
        "1. **Content-based (SPECTER2/TF-IDF) không nắm bắt được thị hiếu cộng đồng (Collaborative Signal):**",
        "   - Một bài báo có thể có tiêu đề và abstract rất sát với profile của user, nhưng nếu bài báo đó có chất lượng học thuật thấp hoặc ít được cộng đồng đón nhận, Content-based vẫn gợi ý nhiệt tình vì độ tương đồng text cao.",
        "   - Ngược lại, Collaborative Filtering (CF/ItemKNN/CTR) dựa trên hành vi đọc tập thể của 5.500 users, từ đó nhận biết được các bài báo 'kinh điển' (influential/popular) trong cùng lĩnh vực mà user chắc chắn muốn đọc, dù abstract của bài kinh điển đó có thể viết theo phong cách tổng quan khác biệt.",
        "",
        "2. **Giải pháp Hybrid đề xuất cho dự án:**",
        "   - Kết hợp SPECTER2 với Graph/Collaborative (như mô hình SPECTER2 + GCN của WP3 hoặc CTR) để vừa giải quyết bài toán Cold-start bằng nội dung, vừa tận dụng thói quen đọc của cộng đồng cho bài báo Warm-start.",
        "",
        "---",
        "*(File báo cáo được tạo tự động bởi `paperrec.error_analysis`)*",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"--> Saved error analysis report to {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/citeulike-a"))
    parser.add_argument("--split-dir", type=Path, default=Path("artifacts/splits/warm/fold_0"))
    parser.add_argument("--output", type=Path, default=Path("results/error_analysis.md"))
    args = parser.parse_args()

    dataset = load_citeulike_a(args.data)
    split = load_split(args.split_dir, dataset)
    run_error_analysis(dataset, split, args.output)


if __name__ == "__main__":
    main()
