import pandas as pd
import ast
import argparse


class TreeNode:
    def __init__(self, position):
        self.position = position          # root, l, r, l_l, r_r_l, ...
        self.left = None
        self.right = None
        self.sequences = []


def parse_sequences(x):
    """
    Convert CSV string like "[1, 2, 3]" into Python list [1, 2, 3].
    """
    if pd.isna(x):
        return []

    if isinstance(x, list):
        return x

    try:
        return ast.literal_eval(str(x))
    except Exception:
        return []


def merge_sequences(seq1, seq2):
    """
    Merge two children's sequences.
    Use dict.fromkeys to remove duplicates while preserving order.
    """
    return list(dict.fromkeys(seq1 + seq2))


def build_tree_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    root = TreeNode("root")

    for _, row in df.iterrows():
        leaf_position = row["leaf_position"]      # example: r_r_l
        sequences = parse_sequences(row["sequences"])

        # A one-node CL warm-start tree has the root itself as its only leaf.
        # The original reconstruction logic assumed every leaf was below the
        # root and therefore rejected the valid position ``root``.
        if str(leaf_position).strip() == "root":
            root.sequences = sequences
            continue

        directions = leaf_position.split("_")

        cur = root
        cur_path = []

        for d in directions:
            cur_path.append(d)
            node_position = "_".join(cur_path)

            if d == "l":
                if cur.left is None:
                    cur.left = TreeNode(node_position)
                cur = cur.left

            elif d == "r":
                if cur.right is None:
                    cur.right = TreeNode(node_position)
                cur = cur.right

            else:
                raise ValueError(f"Invalid direction: {d}")

        # assign leaf sequences
        cur.sequences = sequences

    return root


def dfs_merge(node):
    """
    Postorder DFS:
    1. build left child
    2. build right child
    3. merge them into parent
    """
    if node is None:
        return []

    left_sequences = dfs_merge(node.left)
    right_sequences = dfs_merge(node.right)

    # leaf node
    if node.left is None and node.right is None:
        return node.sequences

    # internal node
    node.sequences = merge_sequences(left_sequences, right_sequences)

    return node.sequences


def collect_nodes(node, rows):
    """
    DFS collect every node into final output table.
    """
    if node is None:
        return

    rows.append({
        "node_position": node.position,
        "sequences": node.sequences
    })

    collect_nodes(node.left, rows)
    collect_nodes(node.right, rows)


def reconstruct_tree_sequences(csv_path, output_path):
    root = build_tree_from_csv(csv_path)

    dfs_merge(root)

    rows = []
    collect_nodes(root, rows)

    out_df = pd.DataFrame(rows, columns=["node_position", "sequences"])
    out_df.to_csv(output_path, index=False)

    return out_df

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct every tree node's sequence membership from leaf summaries."
    )
    parser.add_argument("input_csv", help="Path to sequence_summary.csv")
    parser.add_argument("output_csv", help="Path for tree_node_sequences.csv")
    args = parser.parse_args()
    result = reconstruct_tree_sequences(args.input_csv, args.output_csv)
    print(f"Wrote {len(result)} nodes to {args.output_csv}")


if __name__ == "__main__":
    main()
