# Wake Routing and Retrieval Construction

## 0. 目标与计算链

本 construction 用下面一条统一链路替换 Wake 中逐 sequence、逐 event、
逐 node 的 routing/retrieval 热路径：

\[
\boxed{
\text{Minibatch Prefix Encoding}
\rightarrow
\text{Budgeted Active Frontier}
\rightarrow
\text{Packed Batched Retrieval}
\rightarrow
\text{Vectorized Hawkes Experts}
\rightarrow
\text{Posterior-Guided Coarse-to-Fine Memory Write}
}
\]

它必须同时满足四个约束：

1. **严格因果**：event \(e_{b,t}\) 的预测只能使用
   \(\mathcal H_{b,t}=\{e_{b,j}:j<t\}\)；
2. **prediction 与 teacher 分离**：预测使用 prefix prior \(m^-\)，观察 target
   后得到的 posterior \(q^+\) 只用于训练 Router、更新统计和决定 memory
   ownership；
3. **责任空间与真实计算空间一致**：posterior、MI、balance 和 memory
   ownership 只定义在本次真正参与计算的 frontier/branch 上，不把 coarse
   frontier responsibility 虚构性地投影到未展开 leaves；
4. **计算受预算控制**：每个 prefix 最多使用 \(K_{\max}\) 个 Hawkes
   experts，访问的 tree nodes 和 memory rows 都有确定上界。

当前模型中的 path-additive semantic parameters、node-local episodic banks、
working-memory correction、固定 decay Hawkes basis 以及 Light/Deep Sleep
全部保留。改变的是 Wake 中的组织方式和责任分配方式。

---

## 1. 记号与模型状态

令：

- \(B\)：一个 length bucket 内的 sequence 数量；
- \(T\)：padding 后的最大长度；
- \(N=\sum_{b,t}M_{b,t}\)：minibatch 中有效 target events/prefixes 数；
- \(D\)：event type 数；
- \(M_h\)：Hawkes decay basis 数；
- \(P=D+D^2M_h\)：raw Hawkes parameter 维度；
- \(d_z\)：causal prefix representation 维度；
- \(d_k\)：episodic memory key 维度；
- \(K_{\min}=2,\ K_{\max}=4\)：frontier 最小和最大 expert 数；
- \(R\)：每个 tree node 的 memory capacity；
- \(V_{\max}=2K_{\max}-1=7\)：一次 frontier prediction 最多访问的诱导树节点数。

树中任意 node \(n\) 都是一个合法 coarse-to-fine expert。它的 semantic
raw Hawkes parameters 为

\[
\widetilde\theta_n^{\mathrm{sem}}
=
h_\psi\!\left(\sum_{j\in\operatorname{path}(n)}u_j\right)
+
\sum_{j\in\operatorname{path}(n)}o_j.
\]

因此 internal node 不只是搜索中间状态，也可以直接参加预测、retrieval 和
memory ownership。

---

## 2. Cold-start：用数据相关 residual signature 打破对称性

随机 leaf ID perturbation 不能提供稳定的 specialization signal。首先在共享
cold-start Hawkes parameters \(\theta_0\) 下，为每个训练 sequence \(s\)
计算低维 residual signature：

\[
h_s
=
\mathcal P_r
\left(
-\nabla_{\theta_0}
\mathcal L_{\mathrm{Hawkes}}(s;\theta_0)
\right).
\]

\(\mathcal P_r\) 复用 memory write 使用的 low-rank residual projection。\(h_s\)
表示：为了让共享 Hawkes dynamics 更好地解释 sequence \(s\)，参数应朝哪个
方向修正。

如果数据已经提供 `H_tree` cluster membership，直接以 cluster responsibility
\(q^{(0)}_{s,\ell}\) 计算 leaf residual prototype；否则对
\(\{h_s\}\) 做 balanced \(k\)-means 或 Sinkhorn assignment：

\[
p_\ell^{(0)}
=
\frac{\sum_s q^{(0)}_{s,\ell}h_s}
{\sum_s q^{(0)}_{s,\ell}+\epsilon}.
\]

设目标 leaf mass 为 \(\pi_\ell^{\mathrm{target}}\)，leaf semantic 初始化为

\[
\theta_\ell^{(0)}
=
\theta_0
+
\epsilon_{\mathrm{init}}
\left(
p_\ell^{(0)}
-
\sum_j\pi_j^{\mathrm{target}}p_j^{(0)}
\right),
\]

从而

\[
\sum_\ell
\pi_\ell^{\mathrm{target}}\theta_\ell^{(0)}
\approx
\theta_0.
\]

这保持 cold-start 模型的整体预测，同时给不同方向的 leaves 提供真实、数据
相关的微小差异。internal node 的初始化由 descendant target mass 加权汇总，
而不是重新加入随机扰动。

这一步只在初始化或 topology 发生结构性变化后执行，不属于每个 Wake
minibatch 的热路径。

---

## 3. Minibatch Prefix Encoding

### 3.1 Length bucket 与 padding

将长度相近的 sequences 放入同一 minibatch：

\[
\mathbf t,\mathbf c\in\mathbb R^{B\times T},
\qquad
M\in\{0,1\}^{B\times T},
\]

其中 \(M_{b,t}=1\) 表示有效 target event。

与模型参数无关的 Hawkes statistics 只由 event times、event types 和固定 decay
basis 决定：

\[
X^{\mathrm{hist}},
X^{\mathrm{int}}
\in
\mathbb R^{B\times T\times D\times M_h}.
\]

它们在 dataset cache 中构建一次并跨 epoch 复用。现有
`HawkesBackbone` 的 history/interval cache 直接扩展到 padded minibatch
布局。

### 3.2 一次 recurrent pass 得到全部 strict prefixes

Encoder 对整个 padded minibatch 做一次 recurrent forward：

\[
Z=f_\phi(\mathbf t,\mathbf c)
\in
\mathbb R^{B\times T\times d_z},
\qquad
Z_{b,t}=f_\phi(\mathcal H_{b,t}).
\]

必须保证 row \(t\) 不包含当前 event \(e_{b,t}\)。实现上：

- row \(0\) 使用 trainable `empty_prefix`；
- row \(t>0\) 使用 GRU 在 event \(t-1\) 之后的 hidden state；
- padding positions 由 packed sequence 或 length mask 排除。

随后将所有有效 prefixes flatten：

\[
Z_{\mathrm{flat}}\in\mathbb R^{N\times d_z},
\qquad
N=\sum_{b,t}M_{b,t},
\]

并保留 `flat_to_sequence`、`flat_to_event`，用于将 posterior、loss 和 write
request 映射回原 sequence。

现有 `forward_all_prefix()` 的严格因果定义保留，但接口从单 sequence
\([T]\) 扩展为 padded minibatch \([B,T]\)。

---

## 4. Batched Budgeted Active Frontier

### 4.1 Frontier partition

每个有效 prefix \(i\) 从 root 开始：

\[
\mathcal F_i^{(0)}=\{\mathrm{root}\},
\qquad
m_{i,\mathrm{root}}=1.
\]

展开 internal node \(n\) 时，必须同时加入两个 children：

\[
\mathcal F_i
\leftarrow
\mathcal F_i\setminus\{n\}
\cup
\{n_L,n_R\},
\]

\[
m_{i,c}=m_{i,n}p_{i,c\mid n},
\qquad c\in\{n_L,n_R\}.
\]

因此任意时刻：

\[
\sum_{f\in\mathcal F_i}m_{i,f}=1,
\]

且 frontier 中不存在祖先—后代 pair，其 subtrees 恰好覆盖完整 leaf set。

GPU 上使用固定宽度 tensors：

\[
F^{\mathrm{id}}\in\mathbb N^{N\times K_{\max}},
\quad
F^{\mathrm{mass}}\in\mathbb R^{N\times K_{\max}},
\quad
F^{\mathrm{mask}}\in\{0,1\}^{N\times K_{\max}}.
\]

### 4.2 Neutral structural prior

先定义 node 的 target descendant mass：

\[
M_n^{(0)}
=
\sum_{\ell\in\mathcal L(n)}
\pi_\ell^{\mathrm{target}}.
\]

局部 neutral prior 为

\[
\pi^0_{n\to c}
=
\frac{M_c^{(0)}}
{\sum_{c'\in\operatorname{child}(n)}M_{c'}^{(0)}}.
\]

如果没有数据 mass prior，则令
\(M_n^{(0)}=|\mathcal L(n)|\)。这样最终 leaf-level prior 是 uniform，但不平衡
树的每一层不一定是 \(0.5/0.5\)，从而消除 shallow leaves 的拓扑优势。

neutral prior 只负责结构校准，不是 specialization signal。

### 4.3 Cheap local branch routing

对当前可能展开的 internal node \(n\)，只比较它的两个 children：

\[
\bar z_i=\operatorname{LN}_z(W_zz_i+b_z),
\qquad
\bar e_c=\operatorname{LN}_u(e_c),
\]

\[
x_{i,c}
=
[\bar e_c,\bar z_i,\bar e_c\odot\bar z_i,
|\bar e_c-\bar z_i|],
\]

\[
s^{\mathrm{sem}}_{i,n\to c}
=
\operatorname{MLP}_\rho(x_{i,c}),
\]

\[
a_{i,n\to c}
=
s^{\mathrm{sem}}_{i,n\to c}
+
\lambda_p\log\pi^0_{n\to c},
\]

\[
p_{i,c\mid n}
=
\operatorname{softmax}_c
\left(
\frac{a_{i,n\to c}}{\tau_r}
\right).
\]

训练早期可以保留很小的 prior exploration：

\[
p_{i,c\mid n}
\leftarrow
(1-\epsilon_r)p_{i,c\mid n}
+\epsilon_r\pi^0_{n\to c},
\]

初始建议 \(\tau_r=1.5,\epsilon_r=0.05\)，之后退火到
\(\tau_r=1.0,\epsilon_r=0.01\)。

frontier search 阶段不执行：

- Hawkes NLL；
- episodic memory retrieval；
- 完整 descendant-leaf enumeration。

### 4.4 Expansion utility

局部路由置信度定义为

\[
C_{i,n}
=
1-\frac{H(p_{i,\cdot\mid n})}{\log 2}.
\]

node \(n\) 的历史展开收益由已经完成的 minibatches 更新，而不是偷看当前
target：

\[
G_n
\leftarrow
\rho_GG_n
+
(1-\rho_G)
[\ell_n-\ell_{\mathrm{child}}]_+.
\]

这里 \(\ell_n\) 是 coarse node expert 的 posterior evaluation loss，
\(\ell_{\mathrm{child}}\) 是展开为两个 children 后的 likelihood-mixture loss。
当前 batch 搜索只读取 stop-gradient 的历史 \(G_n\)。

最终 expansion utility：

\[
U_{i,n}
=
\operatorname{sg}(m_{i,n})
\left(G_n+\lambda_cC_{i,n}\right)
-\lambda_{\mathrm{comp}}.
\]

搜索规则：

1. root 非叶时强制展开，保证至少两个 counterfactual experts；
2. 在当前 frontier internal nodes 中选择最大 \(U_{i,n}\)；
3. 当 \(K_i<K_{\min}\) 时继续展开；
4. 当 \(K_i\ge K_{\min}\) 且最大 utility 不为正时停止；
5. 最多保留 \(K_{\max}=4\) 个 experts。

因此每个 prefix 最多执行

\[
K_{\max}-1=3
\]

轮 expansion。每轮对所有 prefixes 的候选 nodes 使用 masked `argmax`、
`gather` 和 `scatter` 一次处理，不在 event loop 中调用 `.item()`。

这里不使用正 entropy bonus。cold-start 时 entropy 最大，正 entropy bonus
会优先扩展最不确定的 branch，并不能表示真实计算收益。

---

## 5. Packed Batched Episodic Retrieval

### 5.1 Persistent bank 与 GPU packed mirror

node-id dictionary 仍作为 topology-aware persistent storage。固定 topology 的
Wake 热路径使用连续 GPU mirror：

\[
K^{\mathrm{mem}}\in
\mathbb R^{|V|\times R\times d_k},
\]

\[
\Delta^{\mathrm{mem}}\in
\mathbb R^{|V|\times R\times P},
\qquad
M^{\mathrm{mem}}\in
\{0,1\}^{|V|\times R}.
\]

同时打包当前 `MemoryBank` 已有状态：

\[
Q^{\mathrm{quality}},Q^{\mathrm{queue}},
U^{\mathrm{usage}},A^{\mathrm{age}}
\in\mathbb R^{|V|\times R}.
\]

packed mirror 在以下时机完整重建：

- 一个 Wake block 开始；
- Light Sleep 完成 residual rebasing 后；
- Deep Sleep 改变 topology、ownership 或 capacity 后。

Wake 中新 write/eviction 直接原位更新对应 node row 和 valid mask。因此新
memory 保持当前实现的在线可见性，不引入“下个 epoch 才能读取”的语义变化。

### 5.2 Path-union gather

对 prefix \(i\) 的 frontier：

\[
\mathcal V_i^{\mathrm{visit}}
=
\bigcup_{v\in\mathcal F_i}
\operatorname{path}(v).
\]

由于 frontier 从 root 通过完整 binary expansion 得到：

\[
|\mathcal V_i^{\mathrm{visit}}|
\le
2K_i-1
\le
V_{\max}=7.
\]

构造 padded visited-node tensors：

\[
V^{\mathrm{id}}\in\mathbb N^{N\times V_{\max}},
\qquad
V^{\mathrm{mask}}\in\{0,1\}^{N\times V_{\max}},
\]

并通过一次 `gather` 得到

\[
K^{\mathrm{visit}}
\in
\mathbb R^{N\times V_{\max}\times R\times d_k}.
\]

共享 ancestors 只出现一次。

### 5.3 Batched sparse retrieval

允许 memory query 依赖当前 prefix 和 visited node：

\[
q_{i,n}=g_\omega(z_i,e_n).
\]

对所有 prefixes、visited nodes 和 memory rows 同时计算：

\[
S_{i,n,r}
=
\gamma\,
\operatorname{cos}(q_{i,n},k_{n,r})
-\lambda_u\log(1+U^{\mathrm{usage}}_{n,r})
-\lambda_aA^{\mathrm{age}}_{n,r}.
\]

invalid rows 置为 \(-\infty\)。为适配当前 `SmoothSparseRetriever`，归一化保留
现有的 sparse support 和 dense exploration floor：

\[
\rho^{\mathrm{sparse}}
=
\operatorname{entmax}_{1.5}(S/\tau_m),
\qquad
\rho^{\mathrm{dense}}
=
\operatorname{softmax}(S/\tau_m),
\]

\[
\rho
=
(1-\epsilon_m)\rho^{\mathrm{sparse}}
+\epsilon_m\rho^{\mathrm{dense}},
\]

当前 retriever 还保留第二层 gated attention：

\[
\alpha_{i,n,r}
=
\frac{
\rho_{i,n,r}
\exp\!\left(
\gamma\operatorname{cos}(q_{i,n},k_{n,r})
\right)
}{
\sum_j
\rho_{i,n,j}
\exp\!\left(
\gamma\operatorname{cos}(q_{i,n},k_{n,j})
\right)
+\epsilon
}.
\]

随后在 valid mask 内重新归一化。write quality 不改变 retrieval support，
而是在 residual aggregation 时缩放 value。top-\(k\)、quality weighting、
usage/staleness 更新、capacity/eviction 和 stable tie ordering 必须与当前
`MemoryBank` 语义保持一致。

node-local correction：

\[
\Delta\widetilde\theta^{\mathrm{node}}_{i,n}
=
\sum_{r=1}^{R}
\alpha_{i,n,r}
Q^{\mathrm{quality}}_{n,r}
\Delta\widetilde\theta_{n,r}^{\mathrm{mem}}.
\]

定义 padded path-incidence tensor：

\[
A^{\mathrm{path}}
\in
\{0,1\}^{N\times K_{\max}\times V_{\max}},
\]

则 frontier correction 为

\[
\Delta\widetilde\theta^{\mathrm{epi}}_{i,k}
=
\sum_n
A^{\mathrm{path}}_{i,k,n}
\Delta\widetilde\theta^{\mathrm{node}}_{i,n}.
\]

热路径只包含 `gather`、batched cosine/matmul、masked normalization 和
`einsum`，不循环每个 sample、node、memory 或 path。

---

## 6. Vectorized Frontier Hawkes Experts

每个有效 frontier candidate 的 raw parameters：

\[
\widetilde\theta_{i,k}
=
\widetilde\theta^{\mathrm{sem}}_{F^{\mathrm{id}}_{i,k}}
+
\Delta\widetilde\theta^{\mathrm{epi}}_{i,k}.
\]

堆叠为

\[
\widetilde\Theta
\in
\mathbb R^{N\times K_{\max}\times P}.
\]

对 raw \(\widetilde\mu,\widetilde W\) 批量应用与当前模型相同的 positivity
transform，再利用缓存的 \(X^{\mathrm{hist}},X^{\mathrm{int}}\) 广播计算全部
\(N\times K_{\max}\) 个 candidate experts 的 target-event NLL：

\[
E_{i,k}
=
\ell^{\mathrm{Hawkes}}
\left(
e_i;\widetilde\theta_{i,k},
X_i^{\mathrm{hist}},
X_i^{\mathrm{int}}
\right),
\]

\[
E\in\mathbb R^{N\times K_{\max}}.
\]

这直接推广当前 `_frontier_sequence_event_nll()`：把单 sequence 的
`[T,K,P]` 扩展为 flattened minibatch 的 `[N,K,P]`，Hawkes 公式和固定
decays 不改变。

---

## 7. Prediction Prior 与 Posterior Teacher 严格分离

### 7.1 Target 前：实际 prediction

在 target event 尚未观察时，frontier prior 是

\[
m^-_{i,k}=F^{\mathrm{mass}}_{i,k}.
\]

实际预测参数沿用当前模型的 parameter mixture：

\[
\widetilde\theta_i^{\mathrm{pred}}
=
\sum_{k=1}^{K_{\max}}
m^-_{i,k}\widetilde\theta_{i,k}
+
\Delta\widetilde\theta_i^{\mathrm{wm}}.
\]

预测损失为

\[
\mathcal L_{\mathrm{pred}}
=
\frac{
\sum_i M_i
\ell^{\mathrm{Hawkes}}
(e_i;\widetilde\theta_i^{\mathrm{pred}})
}{
\sum_i M_i
}.
\]

\(m^-\) 是部署和训练时真实使用的 causal prediction responsibility。

### 7.2 Target 后：evidence posterior

观察到 target 后才能使用 expert NLL \(E_{i,k}\)：

\[
q^+_{i,k}
=
\frac{
m^-_{i,k}\exp(-E_{i,k}/\tau_E)
}{
\sum_j
m^-_{i,j}\exp(-E_{i,j}/\tau_E)
}.
\]

即：

\[
\text{prefix routing prior}
+
\text{current-event likelihood}
\rightarrow
\text{posterior responsibility}.
\]

\(q^+\) 只用于：

- Router teacher；
- branch gain 和 prototype statistics；
- retrieval usage credit；
- memory write ownership 和 quality；
- diagnostics。

它不能回写当前 event 已经完成的预测。

对于 sequence-level diagnostics 或 delayed write window，只在该次实际
frontier \(\mathcal F_i\) 上定义 aggregate evidence：

\[
E^{\mathrm{win}}_{i,k}
=
\frac1{|\mathcal W_i|}
\sum_{e\in\mathcal W_i}
\ell^{\mathrm{Hawkes}}(e;\widetilde\theta_{i,k}),
\]

\[
C^{\mathrm{win}}_{i,k}
=
\operatorname{Compat}
\left(
\bar z_{\mathcal W_i},
u_{F^{\mathrm{id}}_{i,k}}
\right),
\]

\[
A^{\mathrm{win}}_{i,k}
=
-\beta E^{\mathrm{win}}_{i,k}
+
\gamma C^{\mathrm{win}}_{i,k}
+
\log m^-_{i,k}.
\]

这就是 sequence/window 版本的 expert energy、semantic compatibility 和
neutral prior evidence score。它仍然不对未展开的 leaves 定义伪 likelihood。

frontier likelihood mixture：

\[
\mathcal L_{\mathrm{mix}}
=
-\frac1N
\sum_i
\log
\sum_k
m^-_{i,k}\exp(-E_{i,k}/\tau_E).
\]

直接 posterior distillation：

\[
\mathcal L_{\mathrm{post}}
=
\frac1N
\sum_i
D_{\mathrm{KL}}
\left(
\operatorname{sg}[q_i^+]
\;\middle\|\;
m_i^-
\right).
\]

### 7.3 将 frontier posterior 还原为真实 branch teacher

对 prefix \(i\) 中实际被展开的 internal node \(n\)，frontier partition
保证每个 frontier expert 属于 \(n\) 的某一个 child subtree。定义：

\[
\widehat q^+_{i,c\mid n}
=
\frac{
\sum_{k:
F^{\mathrm{id}}_{i,k}\in\operatorname{subtree}(c)}
q^+_{i,k}
}{
\sum_{k:
F^{\mathrm{id}}_{i,k}\in\operatorname{subtree}(n)}
q^+_{i,k}
+\epsilon
}.
\]

这样 Router teacher 只监督真正计算过的 local binary decisions，不需要把
coarse node 的 posterior 投影到它尚未展开的 leaves。

### 7.4 Batch-balanced local Sinkhorn teacher

仅依赖 MI 无法在精确对称点产生第一步 symmetry breaking。对每个实际展开的
node \(n\)，收集本 minibatch 中访问它的样本
\(\mathcal B_n\)，并在它的两个 children 上构造 entropy-regularized
balanced assignment：

\[
Q_n^\star
=
\arg\min_{Q\ge0}
\sum_{i\in\mathcal B_n,c}
Q_{i,c}E^{\mathrm{branch}}_{i,n\to c}
-
\gamma
\sum_{i,c}Q_{i,c}C^{\mathrm{sem}}_{i,n\to c}
-
\tau_QH(Q),
\]

满足

\[
Q\mathbf1=\mathbf1,
\qquad
Q^\top\mathbf1
=
|\mathcal B_n|\pi^0_{n\to\cdot}.
\]

其中
\(C^{\mathrm{sem}}_{i,n\to c}=s^{\mathrm{sem}}_{i,n\to c}\)，child branch
energy 只由该 child subtree 中实际 frontier experts 的 posterior evidence
聚合。先定义条件 prior mass：

\[
m^-_{i,k\mid n}
=
\frac{
m^-_{i,k}
}{
\sum_{j:
F^{\mathrm{id}}_{i,j}\in\operatorname{subtree}(n)}
m^-_{i,j}
+\epsilon
},
\]

再计算：

\[
E^{\mathrm{branch}}_{i,n\to c}
=
-\tau_E
\log
\sum_{k\in\operatorname{subtree}(c)}
m^-_{i,k\mid n}
\exp(-E_{i,k}/\tau_E).
\]

这允许单样本 assignment 很尖锐，同时只约束 minibatch marginal：

\[
\text{sample-level sharp}
+
\text{batch-level balanced}.
\]

Router distillation：

\[
\mathcal L_{\mathrm{dist}}
=
\frac{
\sum_n\sum_{i\in\mathcal B_n}
D_{\mathrm{KL}}
\left(
\operatorname{sg}[Q^\star_{n,i}]
\;\middle\|\;
p_{i,\cdot\mid n}
\right)
}{
\sum_n|\mathcal B_n|
}.
\]

当 \(|\mathcal B_n|\) 太小或 Sinkhorn 不可行时，退化为
\(\widehat q^+_{i,\cdot\mid n}\) teacher。assignment 只在相同 local branch
内平衡，不能把互为祖先/后代的 heterogeneous frontier nodes 当成同一组类别。

### 7.5 Local MI 与 balance

不再对全部 leaves 计算 MI。对访问 node \(n\) 的 minibatch samples：

\[
I_n
=
H\left(
\frac1{|\mathcal B_n|}
\sum_{i\in\mathcal B_n}
p_{i,\cdot\mid n}
\right)
-
\frac1{|\mathcal B_n|}
\sum_{i\in\mathcal B_n}
H(p_{i,\cdot\mid n}).
\]

局部 marginal balance：

\[
\mathcal L_{\mathrm{bal}}
=
\sum_n
w_n
D_{\mathrm{KL}}
\left(
\bar p_n
\;\middle\|\;
\pi^0_{n\to\cdot}
\right).
\]

\(I_n\) 鼓励不同 sequences 作出不同而确定的 branch choice；
\(\mathcal L_{\mathrm{bal}}\) 只约束 batch load，不强迫每个样本都 uniform。
按 node-id 的汇总通过 `scatter_add` 完成，不做 Python grouping。

---

## 8. Posterior-Guided Coarse-to-Fine Memory Write

### 8.1 Delayed causal write request

在 prefix \(i=(b,t)\) 处只创建不包含未来信息的 request snapshot：

\[
r_i^{\mathrm{pre}}
=
\left(
z_i,\ q_i^{\mathrm{mem}},\ \mathcal F_i,\ m_i^-,
\text{event index }t
\right).
\]

当 future horizon
\(\mathcal W_i=\{e_{b,t},\ldots,e_{b,t+h}\}\) 完整可见后，才计算
第 7.2 节的 window-level expert evidence，以及 window posterior：

\[
q^{\mathrm{win}}_{i,k}
=
\operatorname{softmax}_k(A^{\mathrm{win}}_{i,k}).
\]

当 \(\gamma=0,\ \beta=1/\tau_E\) 时，它退化为
\(q^{\mathrm{win}}_{i,k}\propto
m^-_{i,k}\exp(-E^{\mathrm{win}}_{i,k}/\tau_E)\)；若 prefix Router 已经充分
编码 semantic compatibility，默认使用这个无重复计分的形式。

因此 memory ownership 使用真实 future evidence，但不会造成当前 prediction
的 target leakage。

### 8.2 Controller 必须真实控制物理写入

令当前 Controller 的 `MEMORIZE` 和 `QUEUE_SPLIT` probabilities 为
\(p_i^M,p_i^Q\)：

\[
g_i^{\mathrm{write}}=p_i^M+p_i^Q.
\]

候选写入优先级：

\[
\Pi_i
=
g_i^{\mathrm{write}}
\cdot
\operatorname{Conf}(q_i^{\mathrm{win}})
\cdot
[\ell_i^{\mathrm{before}}-\ell_i^{\mathrm{after}}]_+
\cdot
\operatorname{Novelty}(q_i).
\]

只有通过 controller gate 的 requests 才能进入 write staging buffer，并对每条
sequence 使用固定 top-\(B_{\mathrm{write}}\) 上限：

\[
B_{\mathrm{write}}\in[4,8].
\]

`ACCEPT`/`RETRIEVE` 不能再隐式创建 memory。`QUEUE_SPLIT` request 除了写
memory，还保留 queue weight 作为后续 Deep Sleep 的结构证据。

### 8.3 Coarse-to-fine owner

取覆盖 posterior mass \(1-\delta\) 的最小 credible frontier set：

\[
\mathcal C_i(\delta)
=
\operatorname{CredibleSet}
(q_i^{\mathrm{win}},1-\delta).
\]

memory owner 定义为

\[
n_i^{\mathrm{owner}}
=
\begin{cases}
f^\star,
&
\mathcal C_i=\{f^\star\}
\text{ 且 posterior confidence 足够高},
\\[4pt]
\operatorname{LCA}(\mathcal C_i),
&
\text{otherwise}.
\end{cases}
\]

含义是：

- evidence 已能区分到细粒度 frontier node 时，memory 写入该 node；
- 多个方向仍有显著 posterior mass 时，写到它们的共同 coarse ancestor；
- 随着 Router 和 tree 逐渐成熟，同类 memory 的 owner 会自然从 coarse node
  下沉到 specialized descendants；
- 不使用 deterministic leaf-level `argmax`，因此不会把所有 memory 强制压入
  leaf 0；
- 不向未参加计算的 leaves 复制同一 residual。

owner 一定属于本次 path-union，因此已有 semantic parameters、query 和
retrieval context 都可直接复用。

### 8.4 Residual、quality 与 commit

使用与当前 `write_residual_memory()` 一致的 future-window gradient
construction：

\[
\Delta\widetilde\theta_i^{\mathrm{write}}
=
\mathcal P_r
\left(
-\eta
\nabla_{\widetilde\theta_{n_i^{\mathrm{owner}}}^{\mathrm{sem}}}
\mathcal L_{\mathrm{Hawkes}}
(\mathcal W_i)
\right).
\]

写入 item：

\[
m_i
=
\left(
k_i,\,
\Delta\widetilde\theta_i^{\mathrm{write}},\,
\mathcal W_i,\,
q_i^{\mathrm{quality}},\,
p_i^Q
\right),
\]

其中

\[
q_i^{\mathrm{quality}}
\propto
g_i^{\mathrm{write}}
\operatorname{Conf}(q_i^{\mathrm{win}})
[\ell_i^{\mathrm{before}}-\ell_i^{\mathrm{after}}]_+.
\]

同一 minibatch 的 accepted writes 先按 `owner_node_id` 和 slot 批量
partition，再用 tensor index assignment 更新 packed mirror；persistent
dictionary 同步记录 window 和 topology metadata。capacity、eviction、
usage、staleness 以及 stable ordering 继续遵守当前 `MemoryBank` 规则。

---

## 9. Batch Prototype Statistics

data prototype 的在线更新使用 minibatch sufficient statistics，不逐
sequence 调用 Python update。

对实际 frontier/branch posterior 构造 node weight \(W_{s,n}\)，只给真正参与
本 batch 计算的 nodes 分配质量。对 batch representations
\(\bar z_s\)：

\[
C_n=\sum_sW_{s,n},
\]

\[
\mu_n^{\mathrm{batch}}
=
\frac{\sum_sW_{s,n}\bar z_s}
{C_n+\epsilon},
\]

\[
M_{2,n}^{\mathrm{batch}}
=
\sum_sW_{s,n}
(\bar z_s-\mu_n^{\mathrm{batch}})^2.
\]

最后使用一次 parallel Welford merge：

\[
(N_n,\mu_n,M_{2,n})
\oplus
(C_n,\mu_n^{\mathrm{batch}},M_{2,n}^{\mathrm{batch}}).
\]

这与逐 sequence Welford 数学等价，但不再把 active-frontier posterior 投影到
全部 leaves。cold-start `H_tree` membership 只用于第 2 节的初始化 target，
之后由实际 posterior evidence 更新。

---

## 10. 完整 Wake Objective

最终 Wake loss：

\[
\boxed{
\begin{aligned}
\mathcal L_{\mathrm{Wake}}
=\;&
\mathcal L_{\mathrm{pred}}
+
\lambda_{\mathrm{mix}}\mathcal L_{\mathrm{mix}}
+
\lambda_{\mathrm{post}}\mathcal L_{\mathrm{post}}
+
\lambda_{\mathrm{dist}}\mathcal L_{\mathrm{dist}}
\\
&-
\lambda_{\mathrm{MI}}\sum_nw_nI_n
+
\lambda_{\mathrm{bal}}\mathcal L_{\mathrm{bal}}
+
\lambda_{\mathrm{wm}}\mathcal R_{\mathrm{wm}}
+
\lambda_{\mathrm{stab}}\mathcal R_{\mathrm{Hawkes}}.
\end{aligned}
}
\]

职责分离如下：

- \(\mathcal L_{\mathrm{pred}}\)：优化真实部署路径；
- \(\mathcal L_{\mathrm{mix}}\)：让有效 frontier 中至少有 expert 能解释 target；
- \(\mathcal L_{\mathrm{post}}\)：让完整 frontier mass 拟合 event posterior；
- \(\mathcal L_{\mathrm{dist}}\)：在实际展开 branch 上提供 balanced、数据相关的
  teacher；
- local MI：鼓励 sample-specific specialization；
- local balance：控制 batch marginal，防止 load collapse；
- working-memory 和 Hawkes regularizers：沿用当前定义。

建议 warm-up：

1. cold-start 先完成 residual-signature initialization；
2. 前 \(1\sim2\) epochs 使用较高 \(\tau_r,\tau_E,\tau_Q\) 和较小
   \(\lambda_{\mathrm{MI}}\)；
3. Router 能形成稳定 posterior 后逐步增加 distillation/MI 权重；
4. 不需要等待所有 leaves 都被真实展开后才启用训练，因为监督发生在实际
   local branches。

---

## 11. 与当前模型的适配边界

### 11.1 直接复用

- `HawkesBackbone` 的 history/interval statistics cache；
- causal encoder 的 `empty_prefix` 与 `forward_all_prefix()` 定义；
- `TreeSemantics.semantic_theta(node_id)` 的 path-additive semantics；
- `NodeSemanticCompatibility` 的 normalized interaction scorer；
- node-local `MemoryBank` fields、capacity、eviction、usage/staleness；
- `SmoothSparseRetriever` 的 entmax + dense exploration 语义；
- `_frontier_sequence_event_nll()` 的向量化 Hawkes 公式；
- working-memory correction 和 low-rank residual construction；
- Light/Deep Sleep 的 residual rebasing 与 topology transaction。

### 11.2 需要改造

1. `forward_all_prefix()`：增加 padded minibatch 输入；
2. frontier state：从 `list[FrontierSample]` 改为 fixed-width tensors；
3. expansion：按最多 3 轮 GPU masked batch 执行；
4. structural prior：改为 descendant target mass/leaf-count calibration；
5. expansion priority：从正 entropy bonus 改为
   historical gain + confidence - compute cost；
6. episodic hot path：增加 `[node,capacity,*]` packed GPU mirror；
7. retrieval：一次处理全部 prefixes × visited nodes × rows；
8. Hawkes experts：从 `[T,K,P]` 扩展为 `[N,K,P]`；
9. training responsibility：使用 \(q^+\) 和 local branch teacher，不再使用
   full-leaf projection；
10. write request：由 `MEMORIZE/QUEUE_SPLIT` gate 和 per-sequence budget
    真正控制；
11. owner：从 hard leaf `argmax` 改为 posterior credible-set/LCA；
12. prototype update：直接按 actual frontier/node posterior 做 batch
    Welford。

### 11.3 应删除的旧热路径

- event 内逐 node `_route_one()` Python search；
- expansion loop 内 `.item()`；
- 为 MI/balance 构造完整 leaf responsibility；
- 用 event route 的 deterministic leaf `argmax` 决定所有 write target；
- controller action 与物理 write 脱钩的无条件 write request；
- 每个 sample/node/path 单独调用 memory retriever；
- 每条 sequence 重建相同 semantic/static tree tensors。

---

## 12. 复杂度上界

令 \(N\) 为 minibatch 有效 prefixes 数。新 Wake 热路径主要复杂度：

\[
\begin{array}{ll}
\text{Prefix encoder}
&
O(BT\,d_{\mathrm{enc}}),
\\
\text{Frontier search}
&
O(NK_{\max}d_z),
\\
\text{Packed retrieval}
&
O(NV_{\max}R\,d_k),
\\
\text{Hawkes experts}
&
O(NK_{\max}D^2M_h).
\end{array}
\]

其中

\[
K_{\max}=4,
\qquad
V_{\max}=2K_{\max}-1=7.
\]

因此 prediction、retrieval 和 expert evaluation 的上界不再与完整 leaf 数
\(L\) 线性绑定。tree 增长主要影响低频 packed mirror rebuild 和 Sleep，而
不是每个 event 的 GPU hot path。

与旧实现相比，主要收益来自：

- \(B\) 条 sequences 共用一个 recurrent launch；
- 全部 valid prefixes 共用最多 3 轮 frontier kernels；
- path-union memory 通过一个 packed batch 扫描；
- \(N\times K\) Hawkes experts 一次广播计算；
- prototype、MI、balance 和 writes 通过 scatter/index update 汇总；
- 热路径没有逐 event 的 `.cpu()`、`.tolist()` 和 `.item()` 同步。

---

## 13. 必须记录的诊断量

每个 epoch 至少输出：

- `frontier_size_mean/max`；
- `visited_nodes_mean/max`；
- `expansion_rounds_mean/max`；
- `expansion_positive_utility_rate`；
- `branch_prior_mass`、`branch_posterior_mass`；
- `local_route_std`、`local_MI`、`local_balance_KL`；
- `posterior_entropy`、`prior_posterior_KL`；
- `sinkhorn_row_error`、`sinkhorn_marginal_error`；
- `memory_owner_depth_mean`、`owner_lca_rate`；
- `write_candidates`、`write_committed`、`writes_per_sequence`；
- `bank_rows_scanned`、`packed_valid_fraction`；
- `retrieve_effective_k`、`retrieval_similarity`；
- `prefix_encode_ms`、`frontier_ms`、`retrieval_ms`、
  `experts_ms`、`write_ms`。

一个健康的 specialization 状态应满足：

- sample-level posterior 可以尖锐；
- batch-level branch marginal 接近 calibrated prior；
- memory ownership 分布在多个真实参与的 coarse/fine nodes；
- 随训练推进，posterior confidence 和 owner depth 上升；
- `mem_assign` 不再由 leaf 0 的 deterministic tie-break 主导。

---

## 14. 实现验收条件

正式替换旧 Wake routing/retrieval 前必须通过：

1. **因果等价测试**：修改 target/future events 不得改变对应 prefix 的
   \(Z_i,m_i^-\) 和 prediction；
2. **frontier invariant**：无祖先冲突、subtree 完整覆盖、mass 和为 1；
3. **batched-vs-reference**：batched routing/retrieval/Hawkes NLL 与小规模
   Python reference 数值一致；
4. **retrieval 等价**：相同 keys、mask、quality、usage、age 下，packed
   retriever 与当前 `MemoryBank` 输出一致；
5. **online write visibility**：write commit 后 packed mirror 与 persistent
   bank 同步；
6. **posterior support**：\(q^+\) 在 invalid frontier slots 上严格为 0；
7. **no leaf projection**：未展开 leaves 不获得 posterior、MI 或 write credit；
8. **write gate**：`ACCEPT/RETRIEVE` 不创建 write，单 sequence commits 不超过
   \(B_{\mathrm{write}}\)；
9. **gradient test**：Router、semantic offsets、query/retriever、encoder 的
   gradient 都有限且符合 warm-up scale；
10. **performance test**：profile 中不再出现 event-level CUDA sync，GPU
    kernels 以 \([B,T]\)、\([N,K]\) 和 \([N,V,R]\) 为主要 batch shape。

这份 construction 是下一步代码改造的唯一 Wake routing/retrieval 规范；
Sleep 仍通过 packed mirror 的 rebuild/rebase 接口与它衔接。
