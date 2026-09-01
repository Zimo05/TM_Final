"""
DataTree — 层次化聚类树的数据结构模块
============================================

**用途**
--------
本模块为 Hawkes 过程层次化聚类提供一棵二叉树数据结构 (``DataTreeNode``)，
将每个聚类簇中的序列索引关联到树的叶子节点上。内部节点存储其左右子树的
序列索引之并集，从而支持自顶向下快速查询某个子树下覆盖的所有序列。

**数据流**
----------
1. 读取 ``tree_structure.json``（由上游聚类算法生成的树拓扑）和
   ``hawkes_dataset_{k}.csv``（每个序列的 cluster 标签及事件时间戳）。
2. ``process_cluster(k)`` 编排一次完整的构建流程：
   - ``extract_leaf_info`` 从树 JSON 中提取每个叶子节点的位置与估计参数。
   - 按 cluster_id 将序列索引分组，生成 ``sequence_summary`` 表。
   - ``build_data_tree`` 递归构造 ``DataTreeNode`` 树。
   - 断言验证根节点覆盖了数据集中所有序列。
3. 脚本尾部对 ``CLUSTER_COUNTS`` 中的每个 k 批量执行上述流程，
   结果存入全局字典 ``results`` 和 ``data_tree_roots``。

**使用示例**
------------
.. code-block:: python

    from HierarchicalTree.DataTree import process_cluster, CLUSTER_COUNTS

    result = process_cluster(8)              # 处理 8-cluster 的情形
    root: DataTreeNode = result["tree_root"] # DataTreeNode 根节点
    summary: pd.DataFrame = result["sequence_summary"]  # leaf 参数与序列对照表
    leaf_info: list[dict] = result["leaf_info"]         # 叶子节点参数列表

    # 批量结果（模块加载时已自动生成）:
    from HierarchicalTree.DataTree import results, data_tree_roots
    for k, res in results.items():
        print(k, res["total_sequences"])

**依赖**
--------
- Python 3.9+
- pandas
- typing
"""

from typing import Any, Dict, Optional
import json
import pandas as pd

#: 需要处理的聚类数量列表（从 coarse 到 fine）
CLUSTER_COUNTS = [8, 10, 13, 15, 17, 20]

#: 本模块的根目录，用于拼接 ``tree_{k}/`` 子目录路径
BASE_DIR = "/Volumes/shenzm/Shuang_RA/Data"


class DataTreeNode:
    """层次化聚类的二叉树节点。

    每个叶子节点对应一个独立的聚类簇，存储属于该簇的 **所有序列索引**
    (``set[int]``)。内部节点存储其左右子树序列索引的并集，使得树的每一层
    都对应一种粒度的聚类划分。

    Parameters
    ----------
    value : set[int]
        当前节点覆盖的序列索引集合。
    node_type : str
        ``"leaf"`` 表示叶子节点（单一聚类簇），``"internal"`` 表示内部节点。
    left : DataTreeNode | None, optional
        左子节点，叶子节点为 ``None``。
    right : DataTreeNode | None, optional
        右子节点，叶子节点为 ``None``。
    cluster_id : int | None, optional
        叶子节点对应的聚类簇 ID，内部节点为 ``None``。

    Attributes
    ----------
    value : set[int]
    node_type : str
    left : DataTreeNode | None
    right : DataTreeNode | None
    cluster_id : int | None
    """

    def __init__(self, value: set[int], node_type: str,
                 left: Optional["DataTreeNode"] = None,
                 right: Optional["DataTreeNode"] = None,
                 cluster_id: Optional[int] = None):
        self.value = value
        self.node_type = node_type
        self.left = left
        self.right = right
        self.cluster_id = cluster_id

    def __repr__(self):
        """返回便于调试的字符串表示，包含节点类型和序列数量。"""
        n = len(self.value)
        if self.node_type == "leaf":
            return f"Leaf(cluster={self.cluster_id}, {n} seqs)"
        return f"Internal({n} seqs)"


def extract_leaf_info(tree: Dict[str, Any]) -> list[Dict[str, Any]]:
    """从树 JSON 中提取所有叶子节点的位置和 Hawkes 参数。

    通过 DFS 遍历树结构，记录每个叶子节点在树中的路径位置（由 ``"l"``
    和 ``"r"`` 组成，表示从根出发依次选择左/右孩子），以及该聚类簇的
    Hawkes 过程参数。

    Parameters
    ----------
    tree : dict
        表示树拓扑的嵌套字典，格式为::

            {
                "type": "internal" | "leaf",
                "left": { ... },   # 仅 internal
                "right": { ... },  # 仅 internal
                "cluster_id": int, # 仅 leaf
                "params": {
                    "mu": float,   # 背景强度
                    "A": float,    # 激励强度
                    "decay": float # 衰减率
                }
            }

    Returns
    -------
    list[dict]
        叶子节点信息列表，每个元素包含:
            - ``leaf_position`` (str): 节点路径，如 ``"l_r"``
            - ``cluster_id`` (int)
            - ``mu`` (float)
            - ``A`` (float)
            - ``decay`` (float)
    """
    leaves = []

    def dfs(node: Dict[str, Any], path: list[str]):
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


def build_data_tree(node: Dict[str, Any],
                    cluster_to_seqs: Dict[int, set[int]]) -> DataTreeNode:
    """递归构造 ``DataTreeNode`` 树。

    从树 JSON 节点出发，将每个叶子节点关联的 cluster_id 映射为实际序列
    索引集合，内部节点取左右子树的并集，最终构造成完整的二叉树。

    Parameters
    ----------
    node : dict
        树 JSON 中的一个节点（与 ``extract_leaf_info`` 中的格式一致）。
    cluster_to_seqs : dict[int, set[int]]
        cluster_id 到序列索引集合的映射。

    Returns
    -------
    DataTreeNode
        以 ``node`` 为拓扑根、叶子上承载实际序列索引的数据树根节点。
    """
    if node["type"] == "leaf":
        cid = node["cluster_id"]
        return DataTreeNode(value=cluster_to_seqs[cid],
                            node_type="leaf", cluster_id=cid)

    left_child = build_data_tree(node["left"], cluster_to_seqs)
    right_child = build_data_tree(node["right"], cluster_to_seqs)
    merged_value = left_child.value | right_child.value
    return DataTreeNode(value=merged_value, node_type="internal",
                        left=left_child, right=right_child)


def process_cluster(k: int) -> Dict[str, Any]:
    """为给定的聚类数量 ``k`` 编排完整的树构建与数据汇总流程。

    此函数是模块的核心编排入口，依次完成：
        1. 读取 ``hawkes_dataset_{k}.csv`` 和 ``tree_structure.json``
        2. 提取叶子节点参数 (``extract_leaf_info``)
        3. 按 cluster_id 分组序列索引
        4. 生成 ``sequence_summary`` 对照表并持久化为 CSV
        5. 构造 ``DataTreeNode`` 树 (``build_data_tree``)
        6. 断言验证根节点覆盖了所有序列

    Parameters
    ----------
    k : int
        聚类数量，例如 8, 10, 13, 15, 17, 20。

    Returns
    -------
    dict
        包含以下键的字典:
            - ``"k"`` (int): 聚类数量
            - ``"tree_root"`` (DataTreeNode): 构建完成的树根节点
            - ``"sequence_summary"`` (pd.DataFrame):
              列: ``cluster_id``, ``leaf_position``, ``mu``, ``A``,
              ``decay``, ``sequences`` (list[int])。每行是一个叶子聚类簇，
              包含其 Hawkes 参数和该簇下所有序列索引的列表。
            - ``"leaf_info"`` (list[dict]): ``extract_leaf_info`` 的原始输出
            - ``"data_dict"`` (dict[int, list[int]]): cluster_id → 序列索引列表
            - ``"total_sequences"`` (int): 数据集中的总序列数
    """
    tree_dir = f"{BASE_DIR}/tree_{k}"

    # 读取原始数据
    raw_path = f"{tree_dir}/hawkes_dataset_{k}.csv"
    sequences = pd.read_csv(raw_path)

    # 读取树结构
    with open(f"{tree_dir}/tree_structure.json", "r") as f:
        tree = json.load(f)

    # 提取所有 leaf 信息 (position, cluster_id, mu, A, decay)
    leaf_info = extract_leaf_info(tree)

    # cluster_id -> list of sequence indices
    data_dict: Dict[int, list[int]] = {}
    for idx, row in sequences.iterrows():
        cid = int(row["cluster"])
        data_dict.setdefault(cid, []).append(idx)

    # 构建 leaf_info DataFrame，按 cluster_id 对齐 sequences
    leaf_df = pd.DataFrame(leaf_info).sort_values("cluster_id")
    seq_df = pd.DataFrame([
        {"cluster_id": cid, "sequences": seqs}
        for cid, seqs in data_dict.items()
    ]).sort_values("cluster_id")

    # -- Sequence_summary: 将叶子参数与序列列表以 cluster_id 为键合并，
    #    生成一份完整的对照表并保存到 CSV。
    Sequence_summary = leaf_df.merge(seq_df, on="cluster_id")
    Sequence_summary.to_csv(f"{tree_dir}/sequence_summary.csv", index=False)

    # cluster_id -> set of sequence indices
    cluster_to_seqs = {cid: set(seqs) for cid, seqs in data_dict.items()}

    # 构建 DataTree
    data_tree_root = build_data_tree(tree, cluster_to_seqs)

    # 验证：根节点应当包含数据集中的全部序列索引
    all_seq_indices = set(range(len(sequences)))
    assert data_tree_root.value == all_seq_indices, \
        f"tree_{k}: root has {len(data_tree_root.value)} seqs, expected {len(all_seq_indices)}"

    return {
        "k": k,
        "tree_root": data_tree_root,
        "sequence_summary": Sequence_summary,
        "leaf_info": leaf_info,
        "data_dict": data_dict,
        "total_sequences": len(sequences),
    }


# ---------------------------------------------------------------------------
# 模块级批量处理
# ---------------------------------------------------------------------------
# 导入本模块时自动对所有 CLUSTER_COUNTS 构建 DataTree，结果存入以下两个
# 全局字典，供其他模块直接引用，避免重复 I/O 和构建开销。

#: ``results[k]`` → ``process_cluster(k)`` 的完整返回值
results: Dict[int, Dict[str, Any]] = {}
for k in CLUSTER_COUNTS:
    results[k] = process_cluster(k)
    root = results[k]["tree_root"]
    print(f"tree_{k}: root={len(root.value)} seqs, "
          f"left={len(root.left.value)}, right={len(root.right.value)}")

#: ``data_tree_roots[k]`` → 仅 ``DataTreeNode`` 根节点（``results[k]["tree_root"]`` 的快捷方式）
data_tree_roots = {k: r["tree_root"] for k, r in results.items()}
