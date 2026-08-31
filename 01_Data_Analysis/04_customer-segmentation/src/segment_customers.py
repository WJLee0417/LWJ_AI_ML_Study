"""Select, validate, and profile K-Means customer segments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "customer_features.csv"
SEGMENT_PATH = PROJECT_ROOT / "data" / "processed" / "customer_segments.csv"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
REPORT_PATH = PROJECT_ROOT / "results" / "generated" / "segmentation-report.md"
FEATURE_COLUMNS = ["purchase_frequency", "total_revenue", "recency_days", "avg_order_value"]
RANDOM_STATE = 42


def transformed_features(features: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    transformed = features[FEATURE_COLUMNS].copy()
    for column in FEATURE_COLUMNS:
        transformed[column] = transformed[column].clip(lower=0).map(np.log1p)
    scaler = StandardScaler()
    return pd.DataFrame(
        scaler.fit_transform(transformed), columns=FEATURE_COLUMNS, index=features.index
    ), scaler


def select_k(scaled: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    sample = scaled.sample(n=min(20_000, len(scaled)), random_state=RANDOM_STATE)
    rows = []
    for k in range(2, 7):
        model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = model.fit_predict(sample)
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette_score": silhouette_score(sample, labels),
                "smallest_cluster_ratio": pd.Series(labels).value_counts(normalize=True).min(),
            }
        )
    evaluation = pd.DataFrame(rows)
    eligible = evaluation.loc[evaluation["smallest_cluster_ratio"] >= 0.05]
    if eligible.empty:
        raise RuntimeError("No K candidate has a cluster containing at least 5% of the sample.")
    return int(eligible.sort_values("silhouette_score", ascending=False).iloc[0]["k"]), evaluation


def stability_score(scaled: pd.DataFrame, k: int) -> float:
    sample = scaled.sample(n=min(20_000, len(scaled)), random_state=RANDOM_STATE)
    baseline = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(sample)
    scores = []
    for seed in range(10):
        subset = sample.sample(frac=0.8, random_state=seed)
        subset_positions = sample.index.get_indexer(subset.index)
        labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(subset)
        scores.append(adjusted_rand_score(baseline[subset_positions], labels))
    return float(sum(scores) / len(scores))


def save_figures(evaluation: pd.DataFrame, scaled: pd.DataFrame, labels: pd.Series) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 4))
    sns.lineplot(data=evaluation, x="k", y="inertia", marker="o")
    plt.title("Elbow method: inertia by K")
    plt.xticks(evaluation["k"])
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "elbow-score.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.lineplot(data=evaluation, x="k", y="silhouette_score", marker="o", color="#2a7f62")
    plt.title("Silhouette score by K")
    plt.xticks(evaluation["k"])
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "silhouette-score.png", dpi=160)
    plt.close()

    plot_sample = scaled.sample(n=min(20_000, len(scaled)), random_state=RANDOM_STATE)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coordinates = pca.fit_transform(plot_sample)
    plot_labels = labels.loc[plot_sample.index].astype(str)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=coordinates[:, 0], y=coordinates[:, 1], hue=plot_labels, s=12, alpha=0.55)
    plt.title("Customer segments projected with PCA")
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "customer-segments-pca.png", dpi=160)
    plt.close()

    profile = scaled.assign(cluster=labels).groupby("cluster")[FEATURE_COLUMNS].mean()
    plt.figure(figsize=(9, 5))
    sns.heatmap(profile, annot=True, fmt=".2f", center=0, cmap="vlag")
    plt.title("Standardized customer feature profile by cluster")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cluster-profile.png", dpi=160)
    plt.close()


def recommendation(profile: pd.DataFrame, cluster: int) -> str:
    if cluster == profile["total_revenue"].idxmax():
        return "VIP·신상품 우선 안내와 혜택 유지"
    if cluster == profile["recency_days"].idxmax():
        return "재활성화 쿠폰과 선호 카테고리 리마인드"
    if cluster == profile["avg_order_value"].idxmax():
        return "프리미엄 번들·상향 판매 제안"
    return "적립 혜택과 다음 구매 유도 캠페인"


def write_report(features: pd.DataFrame, evaluation: pd.DataFrame, k: int, stability: float) -> None:
    profile = features.groupby("cluster").agg(
        customers=("customer_unique_id", "size"),
        purchase_frequency=("purchase_frequency", "mean"),
        total_revenue=("total_revenue", "mean"),
        recency_days=("recency_days", "mean"),
        avg_order_value=("avg_order_value", "mean"),
    )
    profile["share_pct"] = profile["customers"] / len(features) * 100
    category = (
        features.groupby(["cluster", "preferred_category"]).size().rename("count").reset_index()
        .sort_values(["cluster", "count", "preferred_category"], ascending=[True, False, True])
        .drop_duplicates("cluster")
        .set_index("cluster")["preferred_category"]
    )
    lines = [
        "# 고객 세분화 보고서",
        "",
        "## 군집 수 선택",
        "",
        f"- 선택 K: {k}",
        f"- 안정성: 80% 표본과 seed 10개 평균 ARI = {stability:.3f}",
        "- 선택 규칙: 최소 군집 비율 5% 이상 후보 중 silhouette score가 가장 높은 K",
        "",
        evaluation.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 군집별 특성과 프로모션 제안",
        "",
        "| 군집 | 고객 수 | 비율 | 평균 주문 수 | 평균 누적 매출 | 평균 최근성 일수 | 평균 객단가 | 선호 카테고리 | 제안 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for cluster, row in profile.iterrows():
        lines.append(
            f"| {cluster} | {row['customers']:,.0f} | {row['share_pct']:.1f}% | "
            f"{row['purchase_frequency']:.2f} | {row['total_revenue']:.2f} | "
            f"{row['recency_days']:.1f} | {row['avg_order_value']:.2f} | "
            f"{category.get(cluster, 'unknown')} | {recommendation(profile, cluster)} |"
        )
    lines.extend(
        [
            "",
            "## 한계",
            "",
            "- K-Means는 구형 군집과 유클리드 거리 가정을 사용하므로 다른 군집 구조를 놓칠 수 있다.",
            "- 매출과 최근성은 데이터 기간에 의존하며, 실제 프로모션 전환 효과는 포함하지 않는다.",
            "- 선호 카테고리는 해석용이며 K-Means 입력에는 넣지 않았다.",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not FEATURE_PATH.exists():
        raise FileNotFoundError("Run src/build_features.py before segmentation.")
    features = pd.read_csv(FEATURE_PATH)
    scaled, _ = transformed_features(features)
    k, evaluation = select_k(scaled)
    labels = pd.Series(
        KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit_predict(scaled),
        index=features.index,
        name="cluster",
    )
    stability = stability_score(scaled, k)
    segmented = features.assign(cluster=labels)
    SEGMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    segmented.to_csv(SEGMENT_PATH, index=False, encoding="utf-8")
    save_figures(evaluation, scaled, labels)
    write_report(segmented, evaluation, k, stability)
    print(f"Selected K={k}; mean stability ARI={stability:.3f}")
    print(f"Created {SEGMENT_PATH}")
    print(f"Created figures in {FIGURES_DIR}")
    print(f"Created {REPORT_PATH}")


if __name__ == "__main__":
    main()
