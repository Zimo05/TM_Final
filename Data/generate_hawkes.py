"""
高质量多维 Hawkes 序列生成器
================================

针对原数据三大质量问题的"治本"重写：

1. **谱半径封顶**（消除超长序列爆炸）
   采样激励核 ``A`` 后，按 ``rho(A)`` 整体缩放，强制谱半径 <= ``RHO_CAP``，
   保证过程平稳、事件数可控（原数据部分 cluster rho 高达 0.984，近临界发散）。

2. **固定每条序列的事件数**（替代固定时间窗 -> 长度可控且均衡）
   用 Ogata thinning 模拟，收集满 ``EVENTS_PER_SEQ`` 个事件即停止，
   而不是模拟到固定时间 T。原数据"模拟到 T"导致强激励序列长度爆炸
   （中位数 195，但均值 1432，最长 33821）。

3. **8 类均衡且可预测**
   - mu 背景率各 type 接近且较低，减少随机背景事件
   - 每个 cluster 采样一条覆盖全部 type 的主导事件循环
   - 主导转移之外只保留少量噪声激励
   - 循环是一一映射，因此提升条件可预测性时不会让某一类长期占优

输出
----
- ``tree_{K}/{K}Cluster/THP_{K}.json``  : THP 输入格式
    {"0": [{time_since_start, time_since_last_event, type_event}, ...], "1": [...], ...}
- ``tree_{K}/parameters_{K}.json``      : 各 cluster 的 mu / A / decay 真值参数

用法
----
    python3 generate_hawkes.py            # 用默认配置生成
    python3 generate_hawkes.py --smoke    # 小规模冒烟测试

依赖: numpy
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np

# --------------------------------------------------------------------------- #
# 配置（来自用户确认的治本方案）
# --------------------------------------------------------------------------- #
DIM = 8                  # 事件类型数（8 类）
EVENTS_PER_SEQ = 80      # 每条序列固定事件数（控制长度爆炸的关键）
RHO_MIN = 0.65           # 较少重叠级联时，主导后继更容易识别
RHO_CAP = 0.75           # 严格低于 1，保证过程平稳
SEQS_PER_CLUSTER = 100   # 每个 cluster 生成多少条序列
# 始终写入脚本所在的 Data 目录，避免仓库移动后误写旧绝对路径。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLUSTER_COUNTS = [8, 10, 13, 15, 17, 20]

# mu 背景率：均匀基底（保证 8 类均衡的根基）
MU_BASE = 0.01           # 降低随机背景事件占比
MU_JITTER = 0.002        # 保持各 type 背景率接近

# A 激励结构
DOMINANT_MIN = 0.8       # 主导后继边的未缩放强度范围
DOMINANT_MAX = 1.2
NOISE_SCALE = 0.01       # 非主导边只保留弱噪声
NOISE_DENSITY = 0.10
SELF_SCALE = 0.005       # 避免重复同一 type 成为简单捷径


def sample_parameters(rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """为单个 cluster 采样一组平稳、均衡的 Hawkes 参数。

    Returns
    -------
    dict with keys:
        mu    : (DIM,)      背景强度向量
        A     : (DIM, DIM)  激励核基础矩阵（衰减前）
        decay : float       指数核衰减率
    """
    # --- decay：短记忆，让最近事件的主导后继信号不被旧事件淹没 ---
    decay = float(rng.uniform(8.0, 12.0))

    # --- mu：低且均匀，减少与历史无关的随机背景事件 ---
    mu = MU_BASE + rng.uniform(-MU_JITTER, MU_JITTER, size=DIM)
    mu = np.clip(mu, 0.005, None)

    # --- A：每个 source type 只有一个明显主导的 next type ---
    noise = rng.uniform(0.0, NOISE_SCALE, size=(DIM, DIM))
    noise *= rng.random((DIM, DIM)) < NOISE_DENSITY
    np.fill_diagonal(noise, rng.uniform(0.0, SELF_SCALE, size=DIM))
    A = noise

    # A[target, source] 表示 source 事件对 target 强度的提升。随机 Hamilton
    # cycle 保证每个 type 恰好有一个主导前驱和一个主导后继，避免类别失衡。
    cycle = rng.permutation(DIM)
    successors = np.empty(DIM, dtype=int)
    successors[cycle] = np.roll(cycle, -1)
    dominant_strength = rng.uniform(DOMINANT_MIN, DOMINANT_MAX, size=DIM)
    for source, target in enumerate(successors):
        A[target, source] += dominant_strength[source]

    # 核为 A * decay * exp(-decay*t)，对时间积分后得到 A。因此分支
    # 矩阵就是 A，而不是 A / decay。
    eig = np.abs(np.linalg.eigvals(A))
    rho = float(eig.max())
    if rho > 1e-12:
        target = rng.uniform(RHO_MIN, RHO_CAP)
        A = A * (target / rho)

    return {
        "mu": mu,
        "A": A,
        "decay": decay,
        "successors": successors,
    }


def simulate_sequence(mu: np.ndarray, A: np.ndarray, decay: float,
                      n_events: int, rng: np.random.Generator) -> List[dict]:
    """Ogata thinning 模拟一条多维 Hawkes 序列，收满 ``n_events`` 即停。

    强度: lambda_m(t) = mu_m + sum_{t_i < t} A[m, type_i] * decay * exp(-decay * (t - t_i))

    Returns
    -------
    list[dict]  THP 格式事件列表（time_since_start / time_since_last_event / type_event）
    """
    times: List[float] = []
    types: List[int] = []
    t = 0.0

    # 当前各维强度的"已累积激励"部分（不含 mu），用指数核的可递推性高效更新
    excite = np.zeros(DIM, dtype=float)
    last_t = 0.0

    while len(times) < n_events:
        # 把激励项衰减到当前时刻 t
        if times:
            excite *= np.exp(-decay * (t - last_t))
            last_t = t

        lam = mu + excite                      # 各维当前强度
        lam = np.clip(lam, 0.0, None)
        lam_sum = lam.sum()
        if lam_sum <= 1e-12:
            lam_sum = mu.sum()

        # thinning：以上界 lam_sum 采样候选时间
        w = rng.exponential(1.0 / lam_sum)
        t_cand = t + w

        # 衰减激励到候选时刻，重算真实强度
        decayed = excite * np.exp(-decay * (t_cand - last_t))
        lam_cand = np.clip(mu + decayed, 0.0, None)
        lam_cand_sum = lam_cand.sum()

        if rng.random() <= lam_cand_sum / lam_sum:
            # 接受该事件
            probs = lam_cand / lam_cand_sum
            m = int(rng.choice(DIM, p=probs))
            t = t_cand
            # 更新激励：先衰减到 t，再加上新事件 type m 对各维的激励
            excite = decayed.copy()
            last_t = t
            excite += A[:, m] * decay          # 新事件 m 触发对各维 m' 的激励 A[m', m]
            times.append(t)
            types.append(m)
        else:
            # 拒绝：仅推进时间
            t = t_cand
            excite = decayed
            last_t = t

    # 转 THP 格式
    seq = []
    prev = 0.0
    for tt, mm in zip(times, types):
        seq.append({
            "time_since_start": float(tt),
            "time_since_last_event": float(tt - prev),
            "type_event": int(mm),
        })
        prev = tt
    return seq


def generate_for_k(k: int, rng: np.random.Generator,
                   seqs_per_cluster: int, events_per_seq: int) -> None:
    """为 k-cluster 配置生成全套数据 + 参数文件。"""
    tree_dir = os.path.join(BASE_DIR, f"tree_{k}")
    cluster_dir = os.path.join(tree_dir, f"{k}Cluster")
    os.makedirs(cluster_dir, exist_ok=True)

    # 为每个 cluster 采样参数
    params_record: Dict[str, dict] = {}
    all_sequences: Dict[str, List[dict]] = {}

    seq_global_idx = 0
    for cid in range(k):
        p = sample_parameters(rng)
        params_record[str(cid)] = {
            "mu": p["mu"].tolist(),
            "A": p["A"].tolist(),
            "decay": p["decay"],
            "rho": float(np.abs(np.linalg.eigvals(p["A"])).max()),
            "dominant_successor": p["successors"].tolist(),
        }
        for _ in range(seqs_per_cluster):
            seq = simulate_sequence(p["mu"], p["A"], p["decay"],
                                    events_per_seq, rng)
            all_sequences[str(seq_global_idx)] = seq
            seq_global_idx += 1

    # 保存 THP JSON
    thp_path = os.path.join(cluster_dir, f"THP_{k}.json")
    with open(thp_path, "w") as f:
        json.dump(all_sequences, f)

    # 保存参数真值
    param_path = os.path.join(tree_dir, f"parameters_{k}.json")
    with open(param_path, "w") as f:
        json.dump(params_record, f, indent=2)

    print(f"[tree_{k}] {seq_global_idx} seqs x {events_per_seq} events "
          f"-> {thp_path}")
    print(f"           params -> {param_path}")


def main():
    ap = argparse.ArgumentParser(description="高质量多维 Hawkes 序列生成器")
    ap.add_argument("--smoke", action="store_true",
                    help="小规模冒烟测试（仅 k=8，少量序列）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seqs", type=int, default=SEQS_PER_CLUSTER,
                    help="每个 cluster 的序列数")
    ap.add_argument("--events", type=int, default=EVENTS_PER_SEQ,
                    help="每条序列固定事件数")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.smoke:
        generate_for_k(8, rng, seqs_per_cluster=20, events_per_seq=args.events)
    else:
        for k in CLUSTER_COUNTS:
            generate_for_k(k, rng, seqs_per_cluster=args.seqs,
                           events_per_seq=args.events)


if __name__ == "__main__":
    main()
