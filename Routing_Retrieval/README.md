# Routing Retrieval Investigation

> 新一轮 Wake minibatch 化改造的统一数学与系统规范见
> [`WAKE_ROUTING_RETRIEVAL_CONSTRUCTION.md`](WAKE_ROUTING_RETRIEVAL_CONSTRUCTION.md)。
> 当前代码已经切换到该文档定义的 `posterior_frontier_v2`。

这个目录定义截图里的 **active frontier routing + frontier-only
retrieval** construction。它现在已经通过
`HawkesTree.configure_frontier_routing()` 接入 `Memory/` 主训练与推理版本；
这里仍保留独立 adapter、demo 和 construction 单元测试。

## Construction

对每个样本，frontier 从根开始：

\[
\mathcal F_t^{(0)}=\{\mathrm{root}\},\qquad
\rho_{t,\mathrm{root}}=1.
\]

若展开内部节点 \(n\)，同时用左右两个 child 替换它：

\[
\mathcal F\leftarrow
\mathcal F\setminus\{n\}\cup\{n_L,n_R\},
\qquad
\rho_{t,c}=\rho_{t,n}p_{t,c\mid n}.
\]

所以 frontier 始终满足：

1. 不含互为祖先的节点；
2. 所有 frontier subtree 覆盖整棵树；
3. 每个 leaf 恰好被一个 frontier node 覆盖；
4. \(\sum_{f\in\mathcal F_t}\rho_{t,f}=1\)。

### Local child score

只对 active frontier 中可能被展开节点的两个 children 计算：

\[
s_{t,n\to c}
=
s^{sem}_{t,c}
+\lambda_P\log\pi^0_{n\to c}.
\]

- `semantic_score`：直接复用当前 `NodeSemanticCompatibility(z_t,e_c)`；
- 搜索阶段不读取 target、Hawkes energy、retrieval 或全叶 projection；
- `prior`：把目标 leaf mass 汇总到每个 child 的后代 leaves。默认目标
  leaf mass 均匀，因此不规则树不会偏爱浅叶：

\[
\pi^0_{n\to c}
=
\frac{\sum_{\ell\in Desc(c)}\pi_\ell^{target}}
{\sum_{c'}\sum_{\ell\in Desc(c')}\pi_\ell^{target}}.
\]

局部概率为：

\[
\hat p_{t,c\mid n}
=
\operatorname{softmax}_c(s_{t,n\to c}/\tau_r),
\qquad
p_{t,c\mid n}
=
(1-\epsilon_r)\hat p_{t,c\mid n}
+\epsilon_r\pi^0_{n\to c}.
\]

默认冷启动配置对应截图建议：

- `frontier_budget=4`
- `routing_temperature=1.5`
- `exploration_epsilon=0.05`

这些值可以在训练过程中退火到 `1.0` 和 `0.01`。

### Budgeted best-first expansion

\[
U_t(n)
=
\operatorname{sg}(\rho_{t,n})
\left[G_n+\lambda_C\,Conf(p_{t,\cdot\mid n})\right]
-C_{expand}.
\]

根节点强制展开；达到 `frontier_min_experts` 后，只继续展开正 utility
节点，最多到 `frontier_budget`。默认 \(K_{min}=2,K_{max}=4\)。
`G_n` 由 target 到来后的局部分支 posterior mismatch 做 EMA 更新。

### Frontier-only retrieval

只读取：

\[
\mathcal V_t^{visit}
=
\bigcup_{f\in\mathcal F_t}\operatorname{path}(f).
\]

共享祖先 bank 只调用一次 retrieve。对 frontier node：

\[
\Delta\widetilde\theta^{epi}_{t,f}
=
\sum_{v\in path(f)}
\operatorname{Retrieve}(q_t,\mathcal M_v).
\]

最终参数：

\[
\widetilde\theta_t^{eff}
=
\sum_{f\in\mathcal F_t}
\rho_{t,f}
\left(
\widetilde\theta_f^{sem}
+\Delta\widetilde\theta^{epi}_{t,f}
\right)
+\Delta\widetilde\theta_t^{wm}.
\]

其中 `tree.semantic_theta(f)` 对 internal node 和 leaf 都有效，所以不需要把
frontier 强制下沉到叶子。

## 文件

- `routing_retrieval_investigation/configuration.py`：所有实验超参数；
- `routing_retrieval_investigation/prototype_store.py`：节点数据 prototype；
- `routing_retrieval_investigation/frontier_model.py`：完整 routing、retrieval、
  parameter composition；
- `routing_retrieval_investigation/construction.py`：construction helper；
- `demo.py`：最小运行示例；
- `tests/test_frontier_model.py`：frontier partition、局部计算、受限 retrieval、
  参数合成和梯度验证。

## 最小接入

从仓库根目录运行：

```bash
PYTHONPATH="Memory:Routing Retrieval Investigation" python \
  "Routing Retrieval Investigation/demo.py"
```

训练代码里的接入方式：

```python
from routing_retrieval_investigation import (
    FrontierRoutingConfig,
    FrontierRoutingRetrieval,
)

config = FrontierRoutingConfig(
    frontier_budget=4,
    routing_temperature=1.5,
    exploration_epsilon=0.05,
)
frontier_model = FrontierRoutingRetrieval(tree, config)

output = frontier_model(
    z_t,
    working_delta=working_delta,
    update_memory_state=True,
)
effective = output.effective_params
```

prototype 只由真实 frontier posterior 更新到该节点及其已计算祖先：

```python
frontier_model.prototypes.update_frontier_responsibility(
    prefix_z,
    frontier_node_indices,
    frontier_posterior,
    frontier_mask,
)
```

主训练与推理只使用这一套 active-frontier construction。

## 计算边界

设整棵树有 \(L\) 个 leaves、深度约为 \(H\)，frontier budget 为 \(K_F\)。

- 本 construction 最多展开 \(K_F-1\) 次，只评估 search 中实际进入 active
  frontier 的节点，数量受 \(K_F\) 控制，不随完整 leaf 数线性增长；
- semantic/effective parameter 合成只处理 \(K_F\) 个 experts，复杂度
  \(O(BK_FP)\)；
- retrieval 只访问 frontier paths 的 packed union；binary partition 下实际
  visited tensor 宽度至多 \(2K_F-1\)，共享祖先只读取一次；
- 每个被访问节点内部仍需处理该节点 bank 中的 memory，所以还需要配合
  per-node bank capacity 或 direction summaries，才能同时限制 bank 内部扫描量。

对当前 10-cluster 场景，`K_F=4` 意味着一次 prediction 最多混合 4 个
coarse-to-fine experts，而不是固定计算 10 个 leaves；实际访问 bank 的数量由
这 4 条路径的共享前缀决定。
