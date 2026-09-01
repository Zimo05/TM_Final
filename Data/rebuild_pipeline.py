"""
下游数据链路重建脚本
======================

从 ``generate_hawkes.py`` 已生成的两份"源头"文件出发：
    - tree_{K}/{K}Cluster/THP_{K}.json     （序列，THP 格式）
    - tree_{K}/parameters_{K}.json          （各 cluster 的 mu/A/decay 真值）

自动重建整条下游链路的全部文件：
    1. parameters_{K}.json  追加聚类脚本所需格式（K/d/mu_clusters/A_clusters/decay_factors）
    2. tree_{K}/hawkes_dataset_{K}.csv      （event_times,event_types,cluster）
    3. tree_{K}/tree_structure.json         （层次聚类树，叶子带多维 mu/A/decay）
    4. tree_{K}/dendrogram.png              （聚类树可视化）
    5. tree_{K}/sequence_summary.csv        （leaf_position,cluster_id,mu,A,decay,sequences）
    6. tree_{K}/{K}Cluster/THP_input_{K}.json/.csv （inputConstruct，模型直接输入）

关键约定
--------
序列与 cluster 的对应由生成顺序决定：cluster c 的序列是全局索引
``[c*S, (c+1)*S)``（S = 每 cluster 序列数）。因此 cluster 标签、序列索引
都可精确还原，无需重跑聚类来分配标签。tree_structure.json 的"哪些 cluster
合并"则由各 cluster 参数跑 ward 层次聚类得到。

依赖: numpy, pandas, scipy, scikit-learn, matplotlib
用法:
    python3 rebuild_pipeline.py            # 重建全部 K
    python3 rebuild_pipeline.py --k 8      # 只重建 k=8
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # 无显示环境后端
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from sklearn.preprocessing import StandardScaler

# 与 generate_hawkes.py 使用同一个、可迁移的 Data 根目录。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLUSTER_COUNTS = [8, 10, 13, 15, 17, 20]


# --------------------------------------------------------------------------- #
# 读取源头文件
# --------------------------------------------------------------------------- #
def load_sources(k: int):
    """读取 THP_{K}.json 和 parameters_{K}.json。

    Returns
    -------
    thp : dict[str, list[dict]]   序列（key 为全局索引字符串）
    params : dict[str, dict]      cluster -> {mu, A, decay, rho}
    """
    tree_dir = os.path.join(BASE_DIR, f"tree_{k}")
    with open(os.path.join(tree_dir, f"{k}Cluster", f"THP_{k}.json")) as f:
        thp = json.load(f)
    with open(os.path.join(tree_dir, f"parameters_{k}.json")) as f:
        params = json.load(f)
    return thp, params


def seqs_per_cluster(n_seqs: int, k: int) -> int:
    """每 cluster 序列数（生成时均分）。"""
    assert n_seqs % k == 0, f"序列数 {n_seqs} 不能被 cluster 数 {k} 整除"
    return n_seqs // k


# --------------------------------------------------------------------------- #
# 1. 参数文件：追加聚类脚本所需格式
# --------------------------------------------------------------------------- #
def build_cluster_param_format(params: Dict[str, dict], k: int) -> Dict[str, Any]:
    """把 {cluster:{mu,A,decay}} 转成 {K,d,mu_clusters,A_clusters,decay_factors}。"""
    ordered = [params[str(c)] for c in range(k)]
    mu_clusters = [p["mu"] for p in ordered]            # (K, d)
    A_clusters = [p["A"] for p in ordered]              # (K, d, d)
    decay_factors = [p["decay"] for p in ordered]       # (K,)
    d = len(mu_clusters[0])
    return {
        "K": k,
        "d": d,
        "mu_clusters": mu_clusters,
        "A_clusters": A_clusters,
        "decay_factors": decay_factors,
    }


# --------------------------------------------------------------------------- #
# 2. hawkes_dataset_{K}.csv
# --------------------------------------------------------------------------- #
def build_hawkes_dataset(thp: Dict[str, list], k: int, n_seqs: int) -> pd.DataFrame:
    """从 THP 序列重建 event_times/event_types/cluster 三列 CSV。

    cluster 标签按生成顺序确定：全局索引 idx -> cluster = idx // S。
    """
    S = seqs_per_cluster(n_seqs, k)
    rows = []
    for idx in range(n_seqs):
        seq = thp[str(idx)]
        times = ",".join(f"{e['time_since_start']:.6f}" for e in seq)
        types = ",".join(str(e["type_event"]) for e in seq)
        rows.append({
            "event_times": times,
            "event_types": types,
            "cluster": idx // S,
        })
    return pd.DataFrame(rows, columns=["event_times", "event_types", "cluster"])


# --------------------------------------------------------------------------- #
# 3 & 4. 层次聚类 -> tree_structure.json + dendrogram.png
# --------------------------------------------------------------------------- #
def compute_spectral_radius(A: np.ndarray, decay: float) -> float:
    # generate_hawkes uses A * decay * exp(-decay*t); the integrated kernel
    # is A, so stationarity is governed by rho(A).
    del decay
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def build_theta(cluster_params: Dict[str, Any]) -> np.ndarray:
    """构造用于聚类的特征向量（与 Hierarchical Clustering.py 一致的加权拼接）。"""
    K = cluster_params["K"]
    mu = np.array(cluster_params["mu_clusters"])          # (K, d)
    A = np.array(cluster_params["A_clusters"])            # (K, d, d)
    decay = np.array(cluster_params["decay_factors"]).reshape(K, 1)
    A_flat = A.reshape(K, -1)
    rhos = np.array([compute_spectral_radius(A[c], float(decay[c, 0]))
                     for c in range(K)]).reshape(K, 1)
    w_mu, w_A, w_decay, w_rho = 1.0, 0.5, 0.2, 2.0
    return np.concatenate([w_mu * mu, w_A * A_flat,
                           w_decay * decay, w_rho * rhos], axis=1)


def scipy_tree_to_dict(node) -> Dict[str, Any]:
    if node.is_leaf():
        return {"type": "leaf", "cluster_id": int(node.id)}
    return {"type": "internal",
            "left": scipy_tree_to_dict(node.left),
            "right": scipy_tree_to_dict(node.right)}


def attach_parameters(tree: Dict, cluster_params: Dict[str, Any]) -> Dict:
    mu = cluster_params["mu_clusters"]
    A = cluster_params["A_clusters"]
    decay = cluster_params["decay_factors"]

    def dfs(node):
        if node["type"] == "leaf":
            c = node["cluster_id"]
            node["params"] = {"mu": mu[c], "A": A[c], "decay": decay[c]}
        else:
            dfs(node["left"]); dfs(node["right"])

    dfs(tree)
    return tree


def build_tree_and_dendrogram(cluster_params: Dict[str, Any], k: int,
                              tree_dir: str) -> Dict[str, Any]:
    theta = build_theta(cluster_params)
    theta_scaled = StandardScaler().fit_transform(theta)
    Z = linkage(theta_scaled, method="ward")

    # dendrogram
    fig_path = os.path.join(tree_dir, "dendrogram.png")
    plt.figure(figsize=(10, 6))
    dendrogram(Z, labels=[f"C{i}" for i in range(k)])
    plt.title(f"Hierarchical Clustering of {k} Clusters")
    plt.xlabel("Cluster"); plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # tree
    root, _ = to_tree(Z, rd=True)
    tree = scipy_tree_to_dict(root)
    tree = attach_parameters(tree, cluster_params)
    return tree


# --------------------------------------------------------------------------- #
# 5. sequence_summary.csv
# --------------------------------------------------------------------------- #
def extract_leaf_info(tree: Dict[str, Any]) -> List[Dict[str, Any]]:
    """DFS 提取叶子位置 + 参数（与 DataTree.py 一致）。"""
    leaves = []

    def dfs(node, path):
        if node["type"] == "leaf":
            key = "_".join(path) or "root"
            leaves.append({
                "leaf_position": key,
                "cluster_id": node["cluster_id"],
                "mu": node["params"]["mu"],
                "A": node["params"]["A"],
                "decay": node["params"]["decay"],
            })
            return
        dfs(node["left"], path + ["l"])
        dfs(node["right"], path + ["r"])

    dfs(tree, [])
    return leaves


def build_sequence_summary(tree: Dict[str, Any], k: int,
                           n_seqs: int) -> pd.DataFrame:
    """合并叶子参数与序列索引，列对齐旧格式。"""
    S = seqs_per_cluster(n_seqs, k)
    leaf_df = pd.DataFrame(extract_leaf_info(tree)).sort_values("cluster_id")
    seq_rows = [{"cluster_id": c, "sequences": list(range(c * S, (c + 1) * S))}
                for c in range(k)]
    seq_df = pd.DataFrame(seq_rows).sort_values("cluster_id")
    summary = leaf_df.merge(seq_df, on="cluster_id")
    # 列顺序对齐旧文件: leaf_position,cluster_id,mu,A,decay,sequences
    return summary[["leaf_position", "cluster_id", "mu", "A", "decay", "sequences"]]


# --------------------------------------------------------------------------- #
# 6. THP_input_{K}.json / .csv
# --------------------------------------------------------------------------- #
def build_thp_input(thp: Dict[str, list], k: int, cluster_dir: str) -> None:
    """inputConstruct 等价逻辑：dict -> list，跨平台路径。

    历史数据同时保留 ``.json`` 和 ``.csv`` 两个文件名，但二者实际
    都是相同的 JSON list 格式。两个文件必须一起刷新，否则会混用
    新长度 JSON 与旧长度 ``.csv``。
    """
    input_list = [thp[str(i)] for i in range(len(thp))]
    for extension in ("json", "csv"):
        output_path = os.path.join(
            cluster_dir,
            f"THP_input_{k}.{extension}",
        )
        with open(output_path, "w") as f:
            json.dump(input_list, f, indent=4)


# --------------------------------------------------------------------------- #
# 编排
# --------------------------------------------------------------------------- #
def rebuild_for_k(k: int) -> None:
    tree_dir = os.path.join(BASE_DIR, f"tree_{k}")
    cluster_dir = os.path.join(tree_dir, f"{k}Cluster")

    thp, params = load_sources(k)
    n_seqs = len(thp)

    # 1. 参数文件追加聚类格式
    cluster_params = build_cluster_param_format(params, k)
    merged_params = dict(params)               # 保留 per-cluster 原格式
    merged_params["_cluster_format"] = cluster_params
    with open(os.path.join(tree_dir, f"parameters_{k}.json"), "w") as f:
        json.dump(merged_params, f, indent=2)

    # 2. hawkes_dataset
    hd = build_hawkes_dataset(thp, k, n_seqs)
    hd.to_csv(os.path.join(tree_dir, f"hawkes_dataset_{k}.csv"), index=False)

    # 3+4. tree_structure + dendrogram
    tree = build_tree_and_dendrogram(cluster_params, k, tree_dir)
    with open(os.path.join(tree_dir, "tree_structure.json"), "w") as f:
        json.dump(tree, f, indent=4)

    # 5. sequence_summary
    summary = build_sequence_summary(tree, k, n_seqs)
    summary.to_csv(os.path.join(tree_dir, "sequence_summary.csv"), index=False)
    # 验证根覆盖所有序列
    all_idx = set()
    for seqs in summary["sequences"]:
        all_idx |= set(seqs)
    assert all_idx == set(range(n_seqs)), f"tree_{k}: 序列覆盖不全"

    # 6. THP_input
    build_thp_input(thp, k, cluster_dir)

    print(f"[tree_{k}] OK  seqs={n_seqs}  clusters={k}  "
          f"(hawkes_dataset / tree_structure / dendrogram / "
          f"sequence_summary / THP_input 已更新)")


def main():
    ap = argparse.ArgumentParser(description="下游数据链路重建")
    ap.add_argument("--k", type=int, default=None,
                    help="只重建指定 k；默认全部")
    args = ap.parse_args()
    ks = [args.k] if args.k else CLUSTER_COUNTS
    for k in ks:
        rebuild_for_k(k)


if __name__ == "__main__":
    main()
