# Problem to solve: Hawkes Memory Tree 的问题与目标 Construction

更新日期：2026-07-31

## 0. 文档目的与修改边界

本文档把当前实验暴露的问题、现有代码 construction、根因和目标
construction 按模块对齐。目标不是重写整个模型，而是补齐以下三个尚未
闭合的反馈链：

1. Router specialization：

   \[
   \text{Router}
   \rightarrow
   \text{Router-dependent posterior}
   \rightarrow
   \text{Router}
   \]

2. Controller/write：

   \[
   \text{Controller probability}
   \not\rightarrow
   \text{physical operation}
   \not\rightarrow
   \text{Controller learning}
   \]

3. Memory/Sleep：

   \[
   \text{coarse write}
   \rightarrow
   \text{internal-node bank}
   \not\rightarrow
   \text{Sleep consumer}
   \]

以下部分原则上保持不动：

- Hawkes likelihood 的基本形式；
- Hawkes history/interval cache；
- actual active-frontier，而不是 all-leaf routing；
- frontier budget \(2\le K\le4\)；
- 每个节点都可以拥有 episodic memory 的设计；
- working-memory 的指数衰减形式；
- Split/Merge/Prune 已有的核心 replay likelihood；
- topology mutation 的串行 transaction；
- packed retrieval 的总体接口。

需要修改的是这些模块之间的学习信号、写入 gate、Sleep consumer、统计口径
以及 GPU batch 边界。

---

## 1. 实验现象与已经确认的结论

### 1.1 新 construction 已经改善的部分

相对于最早的实验，当前 posterior-frontier v2 已经完成：

- 使用 descendant target-leaf mass 构造 neutral structural prior；
- 不再把 coarse frontier mass 强制投影到所有 leaves；
- 使用 actual frontier 进行 retrieval、posterior 和 prototype update；
- frontier 搜索与 retrieval 的宽度有界；
- 使用 credible set/LCA 选择 memory owner；
- per-sequence write 数量从几乎每个 event 都写，下降到 top-\(B\)；
- Global optimizer steps 从每 epoch 4 次增加到 16 次；
- soft \(Q\) 不再直接伪造 Deep Sleep pressure。

因此最早的：

```text
mem_assign=[80000, 0, ...]
writes=75000
```

已经不再是当前的主要问题。

### 1.2 当前实验暴露的新问题

| Epoch | writes | M | Q | owner 主要分布 | max observed leaf mass |
|---:|---:|---:|---:|---|---:|
| 1 | 8 | 69 | 0 | root: 1000 | 0.1002 |
| 2 | 745 | 795 | 0 | root: 1000 | 0.1013 |
| 3 | 178 | 205 | 0 | root/node-1: 490/510 | 0.1606 |
| 4 | 26 | 27 | 0 | node-1: 989 | 0.3074 |
| 5 | 7 | 7 | 0 | node-1: 1000 | 0.3923 |

这说明：

1. Leaf 0 collapse 已经被修复，但 collapse 转成了 coarse-branch
   confirmation bias；
2. write 数量由 hard categorical action、候选分布、horizon 和 top-\(B\)
   共同决定，不由真实 residual benefit 单独决定；
3. \(Q=0\)，结构证据生产端没有工作；
4. coarse owner 已出现，但 Light Sleep 仍然只消费 leaves；
5. `sleep_residual=0` 和 `light_absorb=0` 不能证明没有 residual，只能说明
   当前 leaf-only Light Sleep 没有扫描到可用 leaf memories。

### 1.3 当前日志的语义限制

- `owner_hard`：每条 sequence 的 event-majority owner 计数；
- `mem_assign`：所有 events 的 owner 计数，和 event 数相加等于 80000；
- `mem_assign` **不是**实际物理 writes 的 node 分布；
- `max_mass`：只统计实际 frontier 中的 leaf slots，不包含 coarse mass，
  因而不是归一化的全 leaf marginal；
- Global entropy、KL、MI 当前存在重复 sequence weighting，日志绝对值大约
  被放大 63.04 倍；趋势可参考，绝对值不可直接解释。

---

## 2. 必须满足的统一 Invariants

所有模块修改必须同时满足以下约束。

### Invariant 1：teacher 不包含 learned Router prior

\[
q_{\text{teacher}}
\not\ni
p_{\text{router}},
\qquad
q_{\text{teacher}}
\not\ni
m_{\text{frontier}}^{\text{learned}}
\]

固定 topology prior 可以保留，因为它不依赖 Router 输出。

### Invariant 2：Controller gate 直接控制对应 operation

\[
g_A\rightarrow\text{adaptation},\quad
g_R\rightarrow\text{retrieval credit/contribution},\quad
g_W\rightarrow\text{write},\quad
g_Q\rightarrow\text{structural evidence}
\]

hard action 只能用于日志，不能再控制物理操作。

### Invariant 3：每个 write 都存在 Sleep consumer

每条写入最终必须进入以下至少一个分支：

\[
\text{write}
\rightarrow
\{\text{absorb},\text{demote},\text{retain/refine},\text{split}\}
\]

### Invariant 4：未 exposure 的 leaf 是 unknown，不是 zero

\[
exposure_\ell=0
\Rightarrow
mass_\ell=\text{unknown}
\]

unknown leaf 不能积累 low-mass streak，也不能被 prune。

### Invariant 5：连续评价 GPU batch，动态状态边界 commit

\[
\text{packed GPU evaluate}
\rightarrow
\text{masked selection}
\rightarrow
\text{small serial commit}
\]

minibatch 内不 append dynamic banks，不修改 tree topology。

---

## 3. 模块总览

| 模块 | 当前主要问题 | 目标 construction | 覆盖状态 |
|---|---|---|---|
| Encoder/H-tree | 初始表征未对齐，MI 梯度固定 epoch 打开 | alignment stage + reliability-gated routing gradient | 已实现 |
| Router teacher | learned prior 自我强化 | prior-independent local soft-EM | local child teacher 已实现；frontier owner/完整解耦待实现 |
| Router objective | prediction、expert、Router 梯度耦合 | expert/Router/prediction 三路解耦 | 需实现 |
| MI/load | 计算成时间 MI，balance 不能自适应 | branch-local sequence MI + confidence-gated load | 需实现 |
| Expansion | gain 不是 value-of-computation | coarse-vs-expanded prediction gain | 未被共享方案完整覆盖 |
| Controller | 四分类 softmax + physical argmax | 四个独立 sigmoid functional gates | 需实现 |
| Controller learning | 没有 action utility supervision | counterfactual utility BCE | 需实现 |
| Write | proxy gain、无最低阈值 | true residual gain + threshold + top-\(B\) | 需实现 |
| Owner | LCA 正确，但容易长期 coarse | 固定-prior teacher owner + delayed stability | 需实现 |
| Retrieval/rebase | quality 直接缩放 residual | quality-normalized retrieval weights | 需实现 |
| Memory capacity | 热点 node 在 Sleep 前被替换 | per-node admission + diversity-aware replacement | 需实现 |
| Light Sleep | leaf-only | all-node coarse dispatcher | 需实现 |
| Split | \(Q=0\)，internal queue 无 consumer | effective bank queue mass + gradual demotion | 需实现 |
| Prune | unknown leaf 被当成 low mass | exposure-corrected conditional mass | 需实现 |
| Metrics | 指标聚合错误、物理 writes 不可见 | 修正聚合 + operation-level diagnostics | 优先实现 |
| Performance | Wake/event/Sleep Python loops | packed prefix、scatter、batched utility/replay | 分阶段实现 |

---

## 4. Encoder 与 H-tree 表征对齐

### 4.1 当前 construction

当前系统同时包含：

- 离线 Attention/THP pipeline 得到的 H-tree node embeddings；
- 在线训练时创建的 `CausalPrefixEncoder`；
- compatibility MLP，将 \(z_t\) 与 node embedding 拼接后评分；
- Router scorer、prototype 和 Encoder 的在线优化。

compatibility MLP 确实把 \(z_t\) 和 node 放进同一个函数：

\[
a_{t,n}=f_{\text{compat}}([W_z z_t,u_n])
\]

但这不意味着随机初始化的 \(W_z z_t\) 与离线 \(u_n\) 已有共同坐标。
MLP 可以在有监督信号后学习 compatibility，却不能自动提供冷启动对齐。

### 4.2 问题

在 early epochs：

- Encoder、Router scorer、prototype 都不稳定；
- teacher 又依赖 Router prior；
- 因而最初小偏差会被 posterior/distillation 放大；
- 固定 Epoch 3 把 MI 梯度送入 Encoder，会让不可靠分支反向污染
  sequence representation。

### 4.3 目标 construction

#### 方案 A：有离线 sequence/prefix representation 时

若离线 pipeline 能为相同 sequence/prefix 输出 \(z_t^{off}\)，增加投影对齐：

\[
L_{\text{repr-align}}
=
1-\cos(P_{on}z_t^{on},\operatorname{sg}(P_{off}z_t^{off}))
\]

再增加局部分支 distillation：

\[
L_{\text{branch-align}}
=
KL(
\operatorname{sg}(p^{off}_{t,n})
\Vert
p^{on}_{t,n}
)
\]

alignment stage 中：

- 暂停 MI；
- 暂停 prune/merge；
- node embeddings 和 semantic Hawkes parameters 保持固定；
- 只训练 online Encoder projection 和 compatibility scorer；
- 当 held-out branch JS 达到阈值后进入完整训练。

#### 方案 B：没有离线 prefix representation 时

不构造虚假的坐标 target，改用后文的 prior-independent child-energy teacher
作为唯一 alignment signal：

1. 固定 node embeddings；
2. 固定或小学习率更新 semantic experts；
3. 先训练 Router 拟合 child-energy teacher；
4. MI 对 Encoder 的梯度由 reliability gate 连续打开，而不是按 epoch 打开。

#### Encoder routing gradient gate

定义 teacher confidence：

\[
r_n
=
\mathbb E_{s,t:n\ expanded}
\left[
1-\frac{H(q_{s,t,n})}{\log 2}
\right]
\]

定义 teacher-student alignment：

\[
A_n
=
1-\frac{JS(q_n,p_n)}{\log 2}
\]

全局可靠度：

\[
g_{\text{enc}}
=
EMA\left[
\frac{1}{|\mathcal N_{\text{expanded}}|}
\sum_n r_nA_n
\right]
\]

梯度形式：

\[
z_{\text{route}}
=
\operatorname{sg}(z)
+
\alpha_{\max}g_{\text{enc}}
(z-\operatorname{sg}(z)),
\qquad
\alpha_{\max}=0.1
\]

### 4.4 目标验收

- `teacher_student_JS` 连续下降；
- `encoder_route_gate` 从接近 0 平滑上升，而不是 Epoch 3 突然跳到 0.1；
- alignment stage 中 branch assignment 在不同随机 seed 下保持相近；
- 不再出现 MI 刚进入 Encoder 后 owner 迅速从 50/50 翻到 0/1000。

#### 当前实现状态

- 离线 full-leaf membership 已分解为 local binary branch target，并完成独立
  Encoder/compatibility initialization calibration；
- online global training 已使用固定 descendant-mass prior 与 detached
  child-local energy 构造 teacher，不再用 learned Router mass 构造 local
  distillation target；
- `teacher_confidence`、`teacher_student_JS` 和 `encoder_route_gate` 已进入
  epoch 日志；
- Encoder routing gradient 已改为连续 EMA reliability gate，固定 epoch
  warm-up 字段仅为旧 checkpoint/CLI 兼容保留；
- actual-frontier owner teacher、prediction/Router/expert 的完整三路解耦仍属于
  后续模块，不在本轮改动中宣称完成。

---

## 5. Router specialization：prior-independent local soft-EM

### 5.1 当前问题 construction

当前 actual-frontier posterior：

\[
q^+_{t,k}
\propto
m^-_{t,k}
\exp(-E_{t,k}/\tau)
\]

其中 \(m^-\) 是 learned Router 自己产生的 frontier mass。随后 \(q^+\)
又用于：

- Router distillation；
- memory owner；
- prototype update；
- expansion gain update；
- retrieval credit。

闭环：

\[
m_t^-
\rightarrow
q_t^+
\rightarrow
L_{\text{distill}}
\rightarrow
m_{t+1}^-
\]

neutral structural prior 解决了固定 Leaf 0 偏置，但不能消除这个 learned-prior
confirmation loop，因此 collapse 转移到了 branch level。

### 5.2 Local child teacher

对每个在 prefix \(i=(s,t)\) 中实际被展开的 internal node \(n\)，只评价其
两个 immediate children：

\[
c\in\{L,R\}
\]

Router student：

\[
p_{i,n,c}
=
\operatorname{softmax}_c(a_{i,n,c})
\]

child-local energy：

\[
E_{i,n,c}
=
\frac1{|W_i|}
\sum_{\tau\in W_i}
\ell_\tau
\left(
\theta^{sem}_{n,c}
+
\Delta\theta^{epi}_{i,n,c}
\right)
\]

teacher energy 必须满足：

- 不使用 Router mixture；
- 不使用 working-memory residual；
- 只评价当前 child-local semantic + episodic dynamics；
- energy 在构造 teacher 前 detach。

固定 topology prior：

\[
\pi^0_{n,c}
=
\frac{
|\mathcal L(child_c(n))|
}{
|\mathcal L(n)|
}
\]

teacher：

\[
q_{i,n,c}
=
\operatorname{softmax}_c
\left(
\log\pi^0_{n,c}
-
\frac{\operatorname{sg}(E_{i,n,c})}{\tau_q}
\right)
\]

最关键约束：

\[
q_{i,n,c}
\text{ 中不能包含 }
p_{i,n,c}
\text{ 或 learned frontier mass}
\]

### 5.3 Actual-frontier owner teacher

local child teacher 用于 Router 学习；memory owner 需要在实际 frontier cut 上
构造 teacher。

对于 actual frontier \(F_i\)，使用固定 target-leaf structural mass：

\[
\pi^0_{i,k}
=
\frac{
|\mathcal L(node_{i,k})|
}{
\sum_{j\in F_i}|\mathcal L(node_{i,j})|
}
\]

frontier teacher：

\[
q^F_{i,k}
=
\operatorname{softmax}_{k\in F_i}
\left(
\log\pi^0_{i,k}
-
\frac{\operatorname{sg}(E^F_{i,k})}{\tau_F}
\right)
\]

以下操作应使用 \(q^F\)，而不是 learned-mass posterior：

- credible set/LCA owner；
- write owner；
- prototype responsibility；
- actual-frontier retrieval credit target；
- owner diagnostics。

learned Router mass 仍用于：

- 在线选择要展开的 subregion；
- 有界 frontier prediction mixture；
- Router student 输出；

但不能进入 teacher。

### 5.4 Expert、Router 与 prediction 解耦

设 \(m_{i,n}=1\) 表示 node \(n\) 在 prefix \(i\) 中被实际展开。

Expert loss：

\[
L_{\text{expert}}
=
\frac{
\sum_{i,n}m_{i,n}
\sum_c\operatorname{sg}(q_{i,n,c})E_{i,n,c}
}{
\sum_{i,n}m_{i,n}+\epsilon
}
\]

Router distillation：

\[
L_{\text{distill}}
=
-
\frac{
\sum_{i,n}m_{i,n}
\sum_c\operatorname{sg}(q_{i,n,c})
\log p_{i,n,c}
}{
\sum_{i,n}m_{i,n}+\epsilon
}
\]

prediction 仍然使用 learned frontier mass，但在 early/middle persistent
training 中 detach：

\[
\theta_i^{eff}
=
\sum_{k\in F_i}
\operatorname{sg}(p^F_{i,k})
\theta_{i,k}
+
\Delta\theta_i^{wm}
\]

于是：

- prediction NLL 更新 semantic experts、episodic query/retrieval 和 Hawkes
  parameters；
- prediction NLL 不直接更新 Router；
- Router 主要由 \(L_{\text{distill}}\)、MI/load 和 trust region 更新；
- 原 likelihood mixture 保留为监控量，或只给极小 Router gradient。

### 5.5 Prototype update

prototype 不再用 learned-prior posterior 作为 source of truth，改用：

\[
\text{prototype target}
=
\operatorname{sg}(q^F)
\]

只更新 actual frontier nodes；不把 coarse responsibility 投影到 descendants。

### 5.6 目标验收

- `prior_KL` 不再呈指数式增加；
- teacher-student JS 下降；
- root/node-1 owner 不再单调从 50/50 变成 0/1000；
- 不同 seed 的 branch identity 可以置换，但负载和预测质量相近；
- fixed topology prior 下 equal energy 能恢复 descendant target mass。

---

## 6. Branch-local sequence MI、load 与 trust region

### 6.1 当前问题

当前 MI 在每条 sequence 内跨 event prefixes 计算，近似：

\[
I(T;C\mid S=s,n)
\]

它奖励同一 sequence 内不同时间进入不同 children，而不是奖励不同 sequences
形成稳定 specialization。

真正需要：

\[
I(S;C\mid n)
\]

### 6.2 Sequence-first aggregation

对 expanded node \(n\)：

\[
e_{s,n}
=
\sum_t m_{s,t,n}
\]

\[
\bar p_{s,n,c}
=
\frac{
\sum_t m_{s,t,n}p_{s,t,n,c}
}{
e_{s,n}+\epsilon
}
\]

只有 \(e_{s,n}>0\) 的 sequence 才属于 node \(n\) 的统计样本。

node marginal：

\[
\bar p_{n,c}
=
\frac{
\sum_s e_{s,n}\bar p_{s,n,c}
}{
\sum_s e_{s,n}+\epsilon
}
\]

MI：

\[
I(S;C\mid n)
=
H(\bar p_n)
-
\frac{
\sum_s e_{s,n}H(\bar p_{s,n})
}{
\sum_s e_{s,n}+\epsilon
}
\]

### 6.3 Confidence-gated MI 与 load

当两个 child experts 的 energy 几乎相同时，不应该用 MI 随机把 sequences
分开。

teacher confidence：

\[
r_n
=
\frac{
\sum_{i}m_{i,n}
\left(
1-\frac{H(q_{i,n})}{\log2}
\right)
}{
\sum_i m_{i,n}+\epsilon
}
\]

local route regularizer：

\[
\begin{aligned}
L_{\text{route}}
=\;&
L_{\text{distill}}
-\lambda_{MI}\sum_n r_n I(S;C\mid n)\\
&+\lambda_{\text{load}}
\sum_n(1-r_n)
KL(\bar p_n\Vert\pi_n^0)\\
&+\lambda_{TR}L_{TR}
\end{aligned}
\]

trust region：

\[
L_{TR}
=
\sum_{i,n}m_{i,n}
KL(
\operatorname{sg}(p^{EMA}_{i,n})
\Vert
p_{i,n}
)
\]

行为：

- teacher 不确定：用 neutral load 防止随机 collapse；
- teacher 可靠：减弱 balance，允许真实 specialization；
- Router 单 batch 变化过快：EMA trust region 抑制翻转；
- 不需要把 `lambda_route_balance` 提高到产生 uniform collapse。

### 6.4 GPU scatter construction

将有效 prefixes flatten 为 \(N\)，expanded width 为 \(J\le3\)。

复合 segment id：

\[
g_{i,j}
=
sequence\_index_i\cdot N_{node}
+
expanded\_node\_index_{i,j}
\]

第一次 `scatter_add` 聚合 \((sequence,node)\)：

\[
A_{s,n,c}
=
\sum_{i,j:g_{i,j}=(s,n)}
m_{i,j}p_{i,j,c}
\]

\[
C_{s,n}
=
\sum_{i,j:g_{i,j}=(s,n)}
m_{i,j}
\]

第二次按 node 聚合 sequence marginal。整个 MI 不需要 Python
`for sequence` 或 `for node`。

### 6.5 指标聚合修复

batch 内的 entropy/KL/MI 若已经是 sequence sum，epoch accumulator 只能加：

\[
total += batch\_sum
\]

不能再次乘 `sequence_count`。

最终：

\[
epoch\_mean
=
\frac{total\_sum}{total\_sequences}
\]

每个 metric 必须明确自身 reduction：

- `*_sum`：返回 sum；
- `*_mean`：返回 mean；
- epoch accumulator 禁止混用两种口径。

---

## 7. Frontier expansion：真实 value-of-computation

### 7.1 当前问题

当前 expansion gain EMA 更接近 posterior branch mismatch，不等价于展开 children
带来的预测改善。这解释了 frontier 在：

```text
3/5/2 ↔ 4/7/3
```

之间振荡。

### 7.2 目标 construction

对 candidate coarse node \(n\)：

coarse energy：

\[
E_{\text{coarse}}(n)
=
L_W(\theta_n^{sem}+\Delta\theta_n^{epi})
\]

child expanded energy：

\[
E_{\text{expanded}}(n)
=
-
\tau
\log\sum_{c\in\{L,R\}}
\pi_{n,c}^0
\exp(-E_{n,c}/\tau)
\]

value-of-computation：

\[
G_n^{expand}
=
\left[
E_{\text{coarse}}(n)
-
E_{\text{expanded}}(n)
-
C_{\text{expand}}
\right]_+
\]

EMA：

\[
\hat G_n
\leftarrow
\rho_G\hat G_n
+
(1-\rho_G)
\operatorname{sg}(G_n^{expand})
\]

search utility：

\[
U_n
=
\hat G_n
+
\lambda_{conf}C_n
-
\lambda_{cost}C_{\text{compute}}
\]

只有 \(U_n>0\) 且未超过 \(K\) 才展开。这样 frontier width 与实际预测收益相关，
不再由 posterior disagreement 间接决定。

---

## 8. Controller：非互斥 functional gates

### 8.1 当前 construction

当前 `Controller` 已经是 `nn.Module`，也已经进入 checkpoint 和 optimizer；
无需重复完成类的模块化。

真正的问题是当前输出仍为四分类 softmax：

\[
[p_A,p_R,p_M,p_Q]
=
\operatorname{softmax}(logits)
\]

然后：

\[
\arg\max(p)
\in\{M,Q\}
\]

才创建 physical write request。

但真实 operations 是：

- working-memory update；
- retrieval contribution/usage credit；
- episodic write；
- structural evidence。

它们并不互斥。

### 8.2 目标 gates

Controller 输入：

\[
x_i
=
[
\tilde s_i,\,
novelty_i,\,
\log(1+count_i),\,
Conf(q_i^F),\,
H(q_i^F),\,
depth(owner_i)
]
\]

极小 MLP：

\[
[g_A,g_R,g_W,g_Q]
=
\sigma(f_\psi(x_i))
\]

含义：

- \(g_A\)：adaptation strength；
- \(g_R\)：retrieval contribution/credit strength；
- \(g_W\)：write probability；
- \(g_Q\)：在 write 条件下成为结构证据的比例。

### 8.3 实际 operation

Working memory：

\[
\Delta\theta^{wm}_{t+1}
=
\rho\Delta\theta_t^{wm}
-
\eta_f g_{A,t}
\nabla_{\theta^{wm}}\ell_t
\]

retrieval usage credit：

\[
\Delta usage_r
=
g_R
\sum_{k\in F_i}
q^F_{i,k}\alpha_{k,r}
\]

write gate：

\[
g_{\text{write}}=g_W
\]

queue metadata：

\[
queue\_weight=g_Wg_Q
\]

hard enum 可以保留用于日志：

- `QUEUE_SPLIT`：\(g_W\) 和 \(g_Q\) 都高；
- `MEMORIZE`：\(g_W\) 高；
- `RETRIEVE`：\(g_R\) 是主要 gate；
- `ASSIMILATE`：其他情况。

hard enum 不参与 operation。

### 8.4 \(g_R\) 的实施边界

当前系统 retrieval residual 已经在 Controller 计算前参与 prediction。

第一阶段的最小兼容修改：

- \(g_R\) 控制 retrieval usage credit；
- retrieval residual 仍完整参与 prediction；
- 用 with/without retrieval counterfactual 训练 \(g_R\)。

若后续希望 \(g_R\) 也控制 retrieval amplitude，需要将 Controller feature
提前到 prediction composition 之前，并避免用“已经包含 retrieval 的当前
prediction NLL”形成循环。可使用 previous surprise EMA、novelty、count 和
Router uncertainty 作为 pre-prediction feature。

### 8.5 Batch construction

Controller 输入：

\[
X_{ctrl}\in\mathbb R^{N\times d_{ctrl}}
\]

一次 forward：

\[
G\in[0,1]^{N\times4}
\]

hot path 禁止：

```python
float(tensor.cpu())
tensor.item()
argmax().item()
```

hard action 只在 detached batch tensor 上统计。

---

## 9. Controller learning：counterfactual utility

### 9.1 当前问题

当前 `write_penalty=lambda_write*(pM+pQ)` 没有形成完整 Controller
反向路径。即使反向传播，它也只会教 Controller 少写，不会教它何时值得写。

### 9.2 Utility targets

统一标签：

\[
y_j
=
\sigma
\left(
\frac{G_j-C_j}{\tau_u}
\right)
\]

#### Retrieval utility

\[
G_R
=
\left[
L(\theta_{\text{no epi}})
-
L(\theta_{\text{with epi}})
\right]_+
\]

#### Adaptation utility

在后续 window 上：

\[
G_A
=
\left[
L_{t+1:t+h}(\Delta\theta_t^{wm})
-
L_{t+1:t+h}(\Delta\theta_{t+1}^{wm})
\right]_+
\]

#### Write utility

\[
G_W=G_{\text{write}}
\]

具体定义见下一节。

#### Queue/split utility

Sleep 返回实际结构收益：

\[
G_Q
=
\left[
\Delta_{\text{split}}
-
C_{\text{tree}}
\right]_+
\]

产生有效 split 的 queued memories 得到正标签；长期被 retain/reject 的
evidence 得到负标签。

### 9.3 Controller loss

\[
\begin{aligned}
L_{\text{ctrl}}
=\;&
BCE(g_A,\operatorname{sg}(y_A))\\
&+BCE(g_R,\operatorname{sg}(y_R))\\
&+BCE(g_W,\operatorname{sg}(y_W))\\
&+g_WBCE(g_Q,\operatorname{sg}(y_Q))
\end{aligned}
\]

为避免只评价 Controller 自己已经偏好的候选，需要保留少量
\(\epsilon\)-exploration candidates，或对候选采样使用 importance weight。

### 9.4 验收

- `controller_grad_norm > 0`；
- \(g_W\) 与 positive write gain 显著正相关；
- \(g_A\) 与 future adaptation utility 正相关；
- \(g_R\) 与 retrieval counterfactual gain 正相关；
- \(g_Q\) 不再永久为零；
- physical writes 不再依赖 categorical argmax。

---

## 10. Write：真实 residual gain、threshold 与 top-\(B\)

### 10.1 当前问题

当前 priority 的 gain：

\[
E_{\text{mixture}}-\min_kE_k
\]

衡量最佳 frontier expert 相对 mixture 的优势，不是写入 residual 后的收益。
singleton frontier 只能使用人工 \(10^{-6}\) floor。

此外，当前 top-\(B\) 没有最低收益门槛；候选数量少于 \(B\) 时，即使收益接近
零也会被写入。

### 10.2 两阶段 candidate 选择

为了避免对所有 events 计算昂贵 counterfactual：

#### Cheap preselection

\[
\Pi_i^{cheap}
=
g_{W,i}
\cdot novelty_i
\cdot normalized\_surprise_i
\]

每条 sequence 先选择 top-\(C\)，建议：

\[
C=4B
\]

并额外保留少量随机探索候选。

#### Delayed true-gain evaluation

完整 future horizon 到达后，在 owner semantic parameters 上构造 residual：

\[
\Delta\theta_i
=
P_r
\left[
-\eta_m
\nabla_{\theta_{owner}}
L_{W_i}(\theta_{owner})
\right]
\]

这里保持与当前 memory item 的语义一致：

- baseline 使用 owner semantic parameters；
- 不使用 working-memory residual；
- 不使用 Router mixture；
- residual 表示相对于 owner semantic model 的局部修正。

真实收益：

\[
G_i^{write}
=
\left[
L_{W_i}(\theta_{owner})
-
L_{W_i}(\theta_{owner}+\Delta\theta_i)
\right]_+
\]

### 10.3 Owner confidence

不能简单用 \(\max(q^F_i)\) 惩罚 coarse owner，否则合法的 LCA memory 永远
低 priority。

定义 delayed posterior stability：

\[
C_i^{stab}
=
\exp
\left(
-
\frac{
JS(q^{event}_i,q^{window}_i)
}{
\tau_{stab}
}
\right)
\]

owner subtree coverage：

\[
C_i^{owner}
=
\sum_{k:node_{i,k}\in subtree(owner_i)}
q^F_{i,k}
\]

最终 confidence：

\[
C_i
=
C_i^{stab}\cdot C_i^{owner}
\]

这样 leaf 与 coarse owner 都能获得可解释 confidence。

### 10.4 Positive utility filter

\[
\Pi_i
=
g_{W,i}
\cdot C_i
\cdot
[G_i^{write}-C_{\text{mem}}]_+
\cdot novelty_i
\]

候选集合：

\[
\mathcal C_i^+
=
\{
i:
G_i^{write}>C_{\text{mem}}
\land
\Pi_i>\Pi_{\min}
\}
\]

再执行：

\[
selected_s
=
TopB(\mathcal C_s^+)
\]

top-\(B\) 是上限，不是必须填满的 quota。

### 10.5 `write_quality` 映射

当前 `MemoryBank` 要求：

\[
write\_quality\in[0,1]
\]

因此不能直接保存任意尺度的 \(G_i^{write}\)。使用：

\[
write\_quality_i
=
1-\exp
\left(
-
\frac{G_i^{write}}{G_{\text{ref}}}
\right)
\]

queue：

\[
queue\_weight_i
=
g_{W,i}g_{Q,i}
\]

### 10.6 Batched implementation

对当前 minibatch 的 \(R\) 个候选构造：

\[
\Theta_{write}\in\mathbb R^{R\times P},
\qquad
\Theta_{write}.requires\_grad=True
\]

计算每个候选的 future-window loss vector：

\[
L_{win}\in\mathbb R^R
\]

一次：

```python
grad = torch.autograd.grad(
    window_loss.sum(),
    theta_write_batch,
)[0]
```

即可得到 \([R,P]\) 的独立 row gradients，因为每一行参数只影响自己的 loss。

构造 before/after：

\[
\Theta_{candidate}
\in
\mathbb R^{R\times2\times P}
\]

一次 batched NLL 得到：

\[
L_{write}\in\mathbb R^{R\times2}
\]

最后在 padded sequence tensor 上：

```python
priority[invalid_or_negative] = -torch.inf
values, indices = torch.topk(
    priority,
    k=B_write,
    dim=-1,
)
accepted = values > priority_threshold
```

MemoryBank append 只在 minibatch boundary 对 accepted rows 执行。

---

## 11. Owner：保留 credible-set/LCA，但更换 evidence source

### 11.1 保留的部分

credible set/LCA owner 修复了强制 Leaf 0：

1. 按 posterior 降序；
2. 累积到 credible mass；
3. 单节点且 confidence 足够则使用该 node；
4. 否则使用 credible nodes 的 LCA。

这个 construction 不应撤销。

### 11.2 需要修改

owner posterior 从 learned-prior posterior 改为第 5.3 节的固定-prior
actual-frontier teacher \(q^F\)。

write commit 前在完整 horizon 上重新计算 delayed \(q^{F,window}\) 和 owner，
避免用 event-level provisional owner 直接落盘。

### 11.3 必须新增的日志

```text
event_owner_counts
sequence_majority_owner_counts
write_candidates_by_owner
positive_gain_by_owner
physical_writes_by_owner
bank_size_by_node
owner_depth_histogram
owner_lca_rate
```

只有 `physical_writes_by_owner` 才能回答 memory 实际写到哪里。

---

## 12. Episodic retrieval、quality 与 exact rebasing

### 12.1 当前问题

当前 retrieval 的实际 residual：

\[
\delta_{\text{epi}}
=
\sum_i
\alpha_i
\cdot write\_quality_i
\cdot\Delta_i
\]

Light Sleep 当前声称：

\[
\theta^+=\theta+d,
\qquad
\Delta_i^+=\Delta_i-d
\]

是 exact rebasing。但若 `write_quality != 1`：

\[
\sum_i
\alpha_iquality_i(\Delta_i-d)
\ne
\sum_i
\alpha_iquality_i\Delta_i-d
\]

所以 conservation identity 不成立。

### 12.2 目标 retrieval

quality 应影响检索权重，而不是直接缩小 residual amplitude。

定义：

\[
\beta_i
=
\frac{
\alpha_i quality_i
}{
\sum_j\alpha_jquality_j+\epsilon
}
\]

retrieval：

\[
\delta_{\text{epi}}
=
\sum_i\beta_i\Delta_i
\]

当所有有效 quality 都为零时返回 zero retrieval；否则：

\[
\sum_i\beta_i=1
\]

因此：

\[
\begin{aligned}
\theta^+
&=\theta+d\\
\Delta_i^+
&=\Delta_i-d
\end{aligned}
\]

能保证：

\[
\theta^+
+
\sum_i\beta_i\Delta_i^+
=
\theta
+
\sum_i\beta_i\Delta_i
\]

retrieval usage credit 也应更新 \(\beta_i\)，而不是旧的 \(\alpha_i\)。

### 12.3 Demotion residual invariant

coarse memory item 可以被理解为一个 absolute local target：

\[
T_r
=
\theta_n+\Delta_{r,n}
\]

若 memory 从 owner \(n\) demote 到 child \(c\)，保持 item target：

\[
\Delta_{r,c}
=
T_r-\theta_c
=
\theta_n+\Delta_{r,n}-\theta_c
\]

于是：

\[
\theta_c+\Delta_{r,c}
=
\theta_n+\Delta_{r,n}
\]

注意：这保持的是 memory item 的 absolute target，不保证 demotion 前后的
整个 child-path prediction 完全不变。因此 commit 还必须满足：

- child corrected replay gain 为正；
- child corrected model 优于 owner corrected model或差异在容许范围内；
- effective parameter jump 小于 trust bound。

单元测试需要同时验证：

1. item-target conservation；
2. replay gain；
3. demotion 后没有 double-count residual；
4. parent bank 删除该 row，child bank 只出现一次。

---

## 13. Memory capacity 与 admission

### 13.1 当前问题

每个 node bank 容量为 128。Epoch 2 有 745 次 writes，如果大量集中到 root 或
同一个 coarse node，memory 会在 epoch-end Sleep 前被容量 pruning。

当前 capacity score 主要使用 usage、staleness、age 和 fresh bonus，没有：

- true write gain；
- write quality；
- query-key diversity；
- owner 热点保护。

### 13.2 目标 admission

对新候选 \(r\)：

\[
S_r^{admit}
=
\lambda_q quality_r
+
\lambda_u\log(1+usage_r)
+
\lambda_n novelty_r
-
\lambda_a age_r
-
\lambda_s stale_r
\]

若 bank 未满，直接 admit。

若 bank 已满：

1. 找到与新 key 最相似的 existing row；
2. 若 similarity 高于阈值，只保留 gain/quality 更高的一条；
3. 否则与 bank 最低 admission score 比较；
4. 新 row 只有显著更好时才 replacement；
5. 否则 reject，不进行无意义 churn。

### 13.3 Per-node cycle budget

除了 per-sequence top-\(B\)，增加：

\[
writes_{n,cycle}
\le
B_{node}
\]

\(B_{node}\) 可以由：

- bank free capacity；
- owner exposure；
- Sleep replay budget；
- existing residual diversity；

自适应分配。

### 13.4 Micro Light Sleep

当：

\[
new\_writes_n
\ge
\rho_{trigger}\cdot capacity_n
\]

可触发 node-local bounded Light Sleep，避免 epoch 结束前大量高价值 coarse
memories 被丢弃。

### 13.5 日志

```text
write_attempted
write_admitted
write_replaced
write_rejected
evicted_before_sleep
capacity_pressure_by_node
mean_key_diversity_by_node
```

---

## 14. All-node Sleep coarse-memory dispatcher

### 14.1 当前 mismatch

Wake owner support：

\[
\{\text{root},\text{internal nodes},\text{leaves}\}
\]

Light Sleep support：

\[
\{\text{leaves only}\}
\]

所以 coarse bank 有 producer，没有 consumer。

### 14.2 Sleep packed table

将所有 internal-node memory rows 打包：

\[
\begin{aligned}
K_{mem}&\in\mathbb R^{R_c\times d_k}\\
\Delta\Theta_{mem}&\in\mathbb R^{R_c\times P}\\
owner\_index&\in\mathbb N^{R_c}\\
child\_indices&\in\mathbb N^{R_c\times2}
\end{aligned}
\]

对每条 memory 构造：

\[
\begin{aligned}
\Theta_n^{base}&=\theta_n\\
\Theta_n^{mem}&=\theta_n+\Delta_{r,n}\\
\Theta_L^{base}&=\theta_L\\
\Theta_L^{mem}&=\theta_L+\Delta_{r,L}\\
\Theta_R^{base}&=\theta_R\\
\Theta_R^{mem}&=\theta_R+\Delta_{r,R}
\end{aligned}
\]

其中：

\[
\Delta_{r,c}
=
\theta_n+\Delta_{r,n}-\theta_c
\]

一次 replay evaluation 得到：

\[
E_{sleep}\in\mathbb R^{R_c\times6}
\]

### 14.3 Child posterior

child corrected posterior：

\[
q_{r,c}^{sleep}
=
\operatorname{softmax}_c
\left(
\log\pi^0_{n,c}
-
\frac{E(\Theta_c^{mem})}{\tau_{sleep}}
\right)
\]

owner memory gain：

\[
G_{r,n}
=
[E(\Theta_n^{base})-E(\Theta_n^{mem})]_+
\]

child memory gain：

\[
G_{r,c}
=
[E(\Theta_c^{base})-E(\Theta_c^{mem})]_+
\]

### 14.4 三类 consumer

#### A. Demote

若：

\[
\max_cq_{r,c}^{sleep}
\ge
\tau_{demote}
\]

且：

\[
G_{r,c^\star}>G_{\min}^{sleep}
\]

且 child corrected replay 不差于 owner corrected replay超过 margin，则：

- 从 parent bank 删除该 row；
- 使用 rebased \(\Delta_{r,c^\star}\)；
- 写入 immediate child bank；
- 每个 Sleep cycle 最多 demote 一层；
- 保留 write quality、queue weight、usage 和 replay window。

#### B. Shared coarse absorption

当：

- residual cluster direction coherent；
- owner replay gain 稳定为正；
- child posterior 没有单一明显赢家；
- cluster 覆盖多个 child contexts；

则它表示 shared/coarse dynamics，可以吸收到 owner semantic offset。

权重：

\[
\omega_r
=
write\_quality_r
(1+queue\_weight_r)
\]

cluster mean：

\[
\bar\delta_n
=
\frac{
\sum_{r:owner(r)=n}
\omega_r\Delta_r
}{
\sum_{r:owner(r)=n}
\omega_r+\epsilon
}
\]

用小型 grid：

\[
\mathcal A=\{0,0.25,0.5,0.75,1\}
\]

选择：

\[
\alpha_n
=
\arg\min_{\alpha\in\mathcal A}
\left[
\sum_r\omega_r
L(w_r\mid\theta_n+\alpha\bar\delta_n)
+
\lambda_{abs}
\alpha^2\|\bar\delta_n\|^2
\right]
\]

commit：

\[
\theta_n^+
=
\theta_n+\alpha_n\bar\delta_n
\]

\[
\Delta_r^+
=
\Delta_r-\alpha_n\bar\delta_n
\]

在第 12 节的 normalized retrieval construction 下，这是 exact rebasing。

#### C. Retain/refine

若：

- 不能安全 demote；
- 不满足 shared absorption；
- residual 有用但 child evidence 仍不确定或多模态；

则保留在 current owner bank：

- 累积 continuous queue evidence；
- 等待更多同类 memories；
- 下个 Sleep cycle 重新评价；
- 不强制投影到 leaf。

### 14.5 GPU implementation

所有 replay energies、posterior 和 masks 都在 GPU batch 中计算：

\[
M_{demoteL},M_{demoteR},M_{absorb},M_{retain}
\]

按 owner 的 absorption mean 使用 `scatter_add`。只有以下操作保持少量
Python：

- 从 source bank 删除 accepted rows；
- append 到 target child bank；
- 修改 semantic offsets；
- 清理对应 optimizer state。

### 14.6 Sleep 日志

```text
sleep_active_nodes
sleep_active_internal_nodes
coarse_replay_scanned
coarse_residual_energy_by_depth
coarse_demote_count
coarse_absorb_count
coarse_retain_count
coarse_demote_gain
coarse_absorb_gain
```

`sleep_mode` 应区分：

- `light-none`：Light 扫描但没有有效 bank；
- `light-evaluated`：扫描但没有 accepted absorption；
- `light-committed`：发生 absorb/demote；
- `deep-none`；
- `deep-committed`。

---

## 15. Split 与 structural evidence

### 15.1 当前问题

当前 split proposal：

- 只遍历 leaves；
- 依赖外部 `controller.split_queues[leaf_id]`；
- \(Q\) hard action 长期为 0；
- internal-node queue 没有消费路径。

### 15.2 Source of truth

不再用外部整数/浮点 queue 作为 source of truth。对 leaf bank：

\[
Q_\ell^{eff}
=
\sum_{r\in M_\ell}
queue\_weight_r
\cdot write\_quality_r
\]

只有：

\[
Q_\ell^{eff}>Q_{\min}
\]

且有效 replay memories 数量足够，才调用现有 SplitModule。

外部 `split_queues` 可以保留用于旧 checkpoint migration，但不能决定新
construction。

### 15.3 Internal evidence consumer

internal queue evidence 通过 coarse dispatcher：

- child-specific evidence 每个 Sleep demote 一层；
- 最终进入 leaf bank；
- leaf 上的 residual multimodality 才触发 topology split；
- internal node 本身已经有 children，不直接执行 `split_leaf(internal)`。

### 15.4 Split batch boundary

对所有候选 leaves 打包：

\[
\begin{aligned}
residuals&\in\mathbb R^{L_c\times R_{max}\times P}\\
contexts&\in\mathbb R^{L_c\times R_{max}\times d_z}\\
weights&\in\mathbb R^{L_c\times R_{max}}\\
mask&\in\{0,1\}^{L_c\times R_{max}}
\end{aligned}
\]

以下全部 batch evaluate：

- residual distance；
- soft cluster assignment；
- effective mass；
- child prototypes；
- local route probability；
- parent/children replay likelihood；
- split gain。

只有 accepted：

```python
commit_split(...)
```

保持串行，因为需要创建 ParameterDict、banks、optimizer state 和 topology
buffers。

### 15.5 Queue learning feedback

SplitModule 返回：

\[
G_Q=[\Delta_{\text{split}}-C_{\text{tree}}]_+
\]

并将结果回传给产生对应 evidence 的 \(g_Q\) training buffer，闭合：

\[
g_Q
\rightarrow
queue\ evidence
\rightarrow
split\ result
\rightarrow
g_Q
\]

---

## 16. Leaf exposure 与 pruning

### 16.1 当前问题

actual frontier 没有评估某 leaf，不代表该 leaf 概率为零。

当前 leaf responsibility tensor 只包含实际 frontier 中出现的 leaves；若直接
用于 low-mass EMA，unexpanded leaf 会被误当成 low mass。

### 16.2 Exposure statistics

对 leaf \(\ell\)：

\[
exposure_\ell
=
\sum_i
\mathbf 1[\ell\in F_i]
\]

\[
observed\_mass_\ell
=
\sum_i
q^F_{i,\ell}
\mathbf 1[\ell\in F_i]
\]

conditional mass：

\[
\hat m_\ell
=
\frac{
observed\_mass_\ell
}{
exposure_\ell+\epsilon
}
\]

只有 \(exposure_\ell>0\) 时更新：

\[
m_\ell^{EMA}
\leftarrow
\rho m_\ell^{EMA}
+
(1-\rho)\hat m_\ell
\]

没有 exposure：

\[
m_\ell=\text{unknown}
\]

### 16.3 Pruning eligibility

leaf 至少满足：

\[
exposure_\ell^{EMA}\ge e_{\min}
\]

\[
m_\ell^{EMA}<m_{\min}
\]

\[
low\_mass\_streak_\ell\ge patience
\]

并通过现有 replay/survivor evidence，才可 prune。

### 16.4 GPU scatter

现有 frontier tensors：

\[
\begin{aligned}
I&\in\mathbb N^{N\times K}\\
M&\in\{0,1\}^{N\times K}\\
Q&\in\mathbb R^{N\times K}
\end{aligned}
\]

leaf mask：

\[
M_{leaf}
=
M\land isLeaf(I)
\]

两次 scatter：

```python
exposure.scatter_add_(
    0,
    I[M_leaf],
    torch.ones_like(Q[M_leaf]),
)

observed_mass.scatter_add_(
    0,
    I[M_leaf],
    Q[M_leaf],
)
```

### 16.5 结构事务暂停条件

在 Router teacher-student alignment 尚未稳定前：

- 暂停 leaf prune commit；
- 暂停 merge commit；
- 允许 usage consolidation；
- 允许 coarse absorb/demote；
- 允许 stale-memory bookkeeping。

稳定条件不使用固定 epoch，建议同时要求：

\[
EMA(JS(q,p))<\tau_{JS}
\]

\[
g_{\text{enc}}>\tau_{enc}
\]

\[
load\ drift<\tau_{drift}
\]

持续若干 epochs。

---

## 17. Merge/Prune/Split 的 batch-evaluate、serial-commit

现有 Coordinator 的 frozen pre-change snapshot、conflict resolution 和 commit
顺序可以保留。

推荐：

### Merge evaluation

\[
\Theta_{merge}
\in
\mathbb R^{P_m\times V_m\times P}
\]

其中 variants 包含 children、merged parent 和 memory-corrected models。

### Prune evaluation

\[
\Theta_{prune}
\in
\mathbb R^{P_p\times V_p\times P}
\]

批量评价 survivor、removed leaf、collapsed parent 的 replay loss。

### Commit

GPU 上统一得到 proposal scores 和 masks，再执行：

\[
\text{merge}
>
\text{leaf prune}
>
\text{split}
\]

的既有 conflict resolution。只对少量 accepted transactions 串行 commit。

---

## 18. Unified global objective

最终 persistent/global objective：

\[
\begin{aligned}
L_{\text{global}}
=\;&
L_{\text{pred}}(\operatorname{sg}(p))
+
\lambda_E L_{\text{expert}}
+
\lambda_D L_{\text{distill}}\\
&-
\lambda_{MI}
\sum_n r_n I(S;C\mid n)\\
&+
\lambda_{\text{load}}
\sum_n(1-r_n)
KL(\bar p_n\Vert\pi_n^0)\\
&+
\lambda_{TR}L_{TR}
+
\lambda_{\text{ctrl}}L_{\text{ctrl}}
\end{aligned}
\]

Sleep objective 保留 replay/tree/memory/stability 基本项：

\[
L_{\text{sleep}}
=
L_{\text{replay}}
+
\lambda_{\text{tree}}C_{\text{tree}}
+
\lambda_{\text{mem}}C_{\text{mem}}
+
\lambda_{\text{stab}}R_{\text{stab}}
\]

但在 structural proposal evaluation 前加入 coarse-memory dispatcher。

---

## 19. GPU-first Wake construction

### 19.1 Packed prefix layout

minibatch 有 \(B\) 条 sequences：

\[
z\in\mathbb R^{B\times T_{max}\times d_z}
\]

有效 mask：

\[
M_{event}\in\{0,1\}^{B\times T_{max}}
\]

flatten：

\[
N=\sum_{s,t}M_{s,t}^{event}
\]

维护：

\[
sequence\_index\in\mathbb N^N,
\qquad
time\_index\in\mathbb N^N
\]

### 19.2 Frontier layout

\[
\begin{aligned}
frontier\_mass&\in\mathbb R^{N\times K}\\
frontier\_node\_indices&\in\mathbb N^{N\times K}\\
frontier\_mask&\in\{0,1\}^{N\times K}\\
expanded\_node\_indices&\in\mathbb N^{N\times J}\\
expanded\_child\_indices&\in\mathbb N^{N\times J\times2}\\
expanded\_probability&\in\mathbb R^{N\times J\times2}\\
expanded\_mask&\in\{0,1\}^{N\times J}
\end{aligned}
\]

其中：

\[
K\le4,\qquad J\le K-1\le3
\]

### 19.3 推荐 forward

```python
# 1. Encode all valid prefixes.
z_flat, seq_idx, time_idx = encoder.encode_packed(batch)

# 2. Packed active-frontier routing/retrieval.
memory_output = tree(
    z_flat,
    working_delta=wm_delta.index_select(0, seq_idx),
)

# 3. Gather two children for actually expanded nodes.
child_theta, child_prior, expanded_mask = build_child_batch(
    memory_output
)

# 4. Prior-independent batched teacher.
child_energy = batched_window_nll(
    child_theta,
    batch,
    seq_idx,
    time_idx,
)
teacher = softmax(
    log(child_prior)
    - child_energy.detach() / tau_q,
    dim=-1,
)

# 5. Distillation + sequence/node scatter MI.
route_loss = local_route_objective(
    student=memory_output["expanded_probability"],
    teacher=teacher,
    sequence_index=seq_idx,
    node_index=memory_output["expanded_node_indices"],
    mask=expanded_mask,
)

# 6. Batched functional Controller.
features = build_controller_features(...)
gates = controller(features)

# 7. Batched counterfactual utilities.
utilities = compute_batched_utilities(...)
controller_loss = controller_utility_loss(
    gates,
    utilities.detach(),
)

# 8. One shared-parameter backward.
loss = prediction_loss + route_loss + controller_loss
loss.backward()
optimizer.step()

# 9. Threshold + masked per-sequence top-k.
selected = masked_topk_write_candidates(...)

# 10. Boundary commit only.
commit_memory_rows(selected)
```

### 19.4 Hot path 禁止项

```python
tensor.cpu()
tensor.item()
float(tensor)
if tensor:
for memory in bank:
for event in sequence:
torch.autograd.grad(...)  # once per prefix/candidate
```

正确模式：

\[
\text{candidate tensor}
\rightarrow
\text{loss vector}
\rightarrow
\text{one grad(loss.sum())}
\]

### 19.5 Persistent memory mirror

当前 `read_packed()` 每次从动态 banks 重建完整 node-capacity mirror。进一步优化：

- TreeEpisodicMemory 维护 persistent packed mirror；
- bank append/prune/transfer 时只更新 dirty node rows；
- topology mutation 后更新 node-index mapping；
- Wake hot path 只 gather，不重建全部 banks。

### 19.6 当前实现状态（2026-07-31）

已经实现、且不改变状态提交顺序的部分：

1. `CausalPrefixEncoder.forward_padded_prefix()` 对一个 chunk 的
   variable-length sequences 做一次 packed GRU forward；
2. Wake 在 chunk boundary 一次计算 \(z\)、`z_projection`、memory query
   和 packed frontier，随后仍严格按照原 sequence/event 顺序执行
   retrieval、Controller、working-memory update、age 与 write commit；
3. Global 把 batch 内全部有效 prefixes flatten 为 \(N\) 行，只执行一次
   Tree/Router/retrieval/child-teacher forward 和一次 backward；
4. local MI/balance 使用
   \((sequence\_index,node\_index)\) 组合 segment 的 `index_add_` 两级归约，
   保留“node 内平均、sequence 内 node 平均、batch 内 sequence 平均”的
   原始权重；
5. selected top-\(B\) writes 的独立参数行组成
   \(\Theta_{write}\in\mathbb R^{R\times P}\)，一次
   `autograd.grad(window_loss.sum())` 返回全部 row gradients；
6. `TreeEpisodicMemory.read_packed()` 复用 persistent fixed-capacity mirror。
   logical age 只施加 broadcast offset；bank tensor identity/version 变化时
   才刷新 mirror；
7. Global 关闭逐 prefix 的 Python `FrontierSample`/CPU diagnostics
   materialization，仅消费 packed tensors。
8. Sleep 的每个 replay window 使用 cached history/interval tensors 一次
   计算全部 target-event NLL；结构 proposal 仍 batch-evaluate 后串行提交。

为保持模型原始在线语义，以下边界仍串行：

- Wake 的 working-memory recurrence；
- retrieval credit 与 chronological age commit；
- hard Controller action、credible-set/LCA owner；
- episodic bank append；
- split/merge/prune topology transaction commit。

因此“batch 化”指相同输入状态下的纯计算合并，并不把未来 memory state
提前提供给较早事件。

---

## 20. Working memory batch

单 sequence inference 保持 \(B=1\)；训练时：

\[
\Delta\Theta^{wm}
\in
\mathbb R^{B\times P}
\]

更新：

\[
\Delta\Theta_{t+1}^{wm}
=
\rho\Delta\Theta_t^{wm}
-
\eta_f
g_{A,t}\odot
\nabla_{\Theta_t^{wm}}L_t
\]

代码：

```python
wm_delta.mul_(rho)
wm_delta.add_(
    -eta_fast
    * adapt_gate.unsqueeze(-1)
    * wm_grad
)
wm_delta.mul_(active_sequence_mask.unsqueeze(-1))
```

sequence 完成后 mask reset；不为训练和 inference 保留两套数学逻辑。

---

## 21. 文件级修改映射

### `Routing_Retrieval_Investigation/.../frontier_model.py`

保留：

- fixed-width best-first expansion；
- target-leaf structural prior；
- packed actual frontier；
- \(K\le4\)。

修改/新增：

- 对外暴露 fixed actual-frontier prior；
- 保证 expanded child indices/prior 可直接用于 local teacher；
- expansion EMA 改为真实 value-of-computation；
- persistent packed memory mirror 的 dirty-row 接口由 memory 模块配合。

### `Memory/Train/Train.py`

修改：

- `_frontier_posterior` 分离 learned prediction posterior 与 fixed-prior teacher；
- actual-frontier owner 使用 fixed-prior teacher；
- `_local_frontier_objective` 改为 branch-local sequence MI；
- prediction、expert、Router loss 解耦；
- reliability-gated Encoder routing gradient；
- Controller utility loss；
- batched true write gain；
- threshold + masked top-\(B\)；
- Global metric aggregation修复；
- physical writes-by-owner diagnostics；
- Sleep 接收 exposure stats 与 all-node memory evidence。

### `Memory/Wake/SequentialController.py`

当前已经是 `nn.Module`，保留 checkpoint/state 接口。

修改：

- 四分类 softmax 改为四个 independent sigmoid gates；
- Controller 接收 batched feature tensor；
- 删除 hard argmax 对 operation 的控制；
- hard enum 只保留日志；
- 输出 `adapt/retrieve/write/queue_given_write`；
- utility supervision 由 Train 提供。

### `Memory/Train/Inference.py`

修改：

- 使用与 Train 相同的 functional gates；
- write 使用 delayed true residual gain；
- threshold 后 top-\(B\)；
- queue weight 使用 \(g_Wg_Q\)；
- owner 使用 fixed-prior frontier teacher；
- checkpoint 迁移旧 softmax Controller 参数时显式标记版本。

### `Memory/MemoryResiduals/MemoryBank.py`

当前已经包含 `write_quality`、`queue_weight`。

修改：

- quality 进入 normalized retrieval weight，不直接缩放 residual amplitude；
- capacity admission 使用 gain/quality/diversity；
- 记录 admission/replacement/rejection；
- 提供批量 row transfer/keep 接口；
- exact rebase tests。

### `Memory/MemoryResiduals/EpisodicMemory.py`

修改：

- `read_packed` 返回 normalized quality-aware weights；
- retrieval credit 使用同一 normalized weights；
- persistent node-capacity mirror；
- dirty-node synchronization；
- coarse row transfer 接口。

### `Memory/Sleep/Light.py`

修改：

- leaf-only active set 改为 all active nodes；
- coarse dispatcher 的 absorb/demote/retain evaluation；
- batched replay；
- internal-node semantic absorption；
- 按 depth 输出 residual energy；
- Light status 细分。

### `Memory/Sleep/Coordinator.py`

Sleep 顺序调整为：

1. usage consolidation；
2. coarse dispatcher；
3. absorb/demote/retain commit；
4. exposure mass update；
5. build leaf split proposals；
6. batch evaluate split/merge/prune；
7. conflict resolution；
8. serial topology commit；
9. stale-memory prune。

### `Memory/Sleep/Split.py`

保留核心 replay likelihood。

修改：

- source weight 使用 `write_quality * queue_weight`；
- 支持 leading leaf-batch dimension；
- 返回可用于 \(G_Q\) supervision 的 split benefit；
- source of truth 从 external queue 改为 bank evidence。

### `Memory/Sleep/Prune.py`

修改：

- 输入 exposure-corrected conditional mass；
- exposure=0 时 unknown；
- exposure EMA 低于门槛时禁止 prune；
- proposal evaluation batch 化。

### `Memory/Sleep/Merge.py`

核心 merge score 与 commit 保持不动。

只新增：

- batch proposal evaluation adapter；
- 与 reliability gate 配合暂停 commit；
- 复用 residual rebase helper。

---

## 22. 分阶段实施顺序

### Phase 0：先修诊断，不改变模型行为

1. 修正 Global metric aggregation；
2. 新增 physical writes-by-owner；
3. 新增 bank size/capacity churn；
4. 新增 Light Sleep scanned nodes/depth；
5. 新增 leaf exposure；
6. 区分 `light-evaluated` 与 `deep-none`。

验收：现有 run 的所有关键计数都能明确解释。

### Phase 1：Router specialization

1. fixed-prior local child teacher；
2. fixed-prior actual-frontier owner teacher；
3. prediction/expert/Router gradient 解耦；
4. sequence/node scatter MI；
5. confidence-gated load；
6. trust region；
7. reliability-gated Encoder gradient；
8. true expansion gain。

验收：不再 coarse branch collapse，再进入下一阶段。

### Phase 2：Controller 与 write

1. independent gates；
2. gates 控制 actual operations；
3. batched counterfactual utilities；
4. Controller loss；
5. true residual gain；
6. threshold + masked top-\(B\)；
7. per-node admission。

验收：Controller gradient、gate/utility correlation 和 write quality 达标。

### Phase 3：Retrieval/rebase 与 all-node Sleep

1. quality-normalized retrieval；
2. exact rebase tests；
3. coarse dispatcher；
4. absorb/demote/retain；
5. internal queue evidence 逐层进入 leaf；
6. effective bank queue trigger；
7. Controller \(g_Q\) 的 Sleep feedback。

验收：coarse memories 确实被扫描和消费。

### Phase 4：Exposure-safe structure learning

1. exposure-corrected mass；
2. reliability-gated prune/merge；
3. batch proposal evaluation；
4. serial topology commit；
5. structural regression tests。

在 Phase 1–3 未完成前，不应启用正式 leaf pruning。

### Phase 5：完整 GPU hot-path 优化

1. packed prefix Encoder；
2. batched Hawkes child/window NLL；
3. batched Controller utilities；
4. batched working memory；
5. persistent memory mirror；
6. batched Sleep replay；
7. batch-evaluate/serial-commit topology transactions。

---

## 23. 必须新增的测试

### Router

1. equal child energy 恢复 fixed descendant prior；
2. teacher 对 Router logits 的梯度为零；
3. prediction loss 在 detach 阶段不更新 Router；
4. expert loss 不通过 teacher energy target 反向修改 teacher；
5. sequence MI 对 sequence permutation invariant；
6. 同一 sequence 内时间翻转不能单独提高 \(I(S;C\mid n)\)；
7. low teacher confidence 时 load KL 生效；
8. high confidence 时允许 specialization；
9. expansion gain 等于 coarse-vs-expanded NLL improvement。

### Controller/write

1. 即使 hard diagnostic action 为 A，高 \(g_W\)+正 utility 仍可写；
2. 即使存在空余 top-\(B\) slot，负 utility 不写；
3. writes 每 sequence 不超过 \(B\)；
4. \(g_Q>0\) 的 M-like write 仍保存 queue evidence；
5. Controller parameters 获得 finite nonzero gradient；
6. batched write gain 与逐 candidate reference 一致；
7. incomplete horizon 不被提前提交。

### Memory/rebase

1. normalized retrieval weights 和为 1；
2. semantic absorption 前后 effective parameters 守恒；
3. internal-node absorption 对 descendant path 不 double-count；
4. demotion item target 守恒；
5. demotion source row 被删除、target row 只出现一次；
6. quality 很小时不产生 `shift/quality` 数值爆炸；
7. capacity admission 保留高 gain、多样 memories。

### Sleep

1. internal/root bank 能被 Light Sleep 扫描；
2. child-specific memory demote 一层；
3. shared coarse cluster absorb 到 owner；
4. ambiguous memory retain；
5. internal queue evidence 最终能进入 leaf split；
6. `sleep_residual` 包含 coarse bank energy；
7. batch replay 与 reference replay 一致。

### Prune

1. exposure=0 的 leaf 不更新 low-mass streak；
2. low conditional mass 但 exposure 不足时禁止 prune；
3. exposure 足够且连续 low mass 时才进入 proposal；
4. packed scatter 与 reference counting 一致。

### Performance

1. Wake hot path 不出现 per-prefix `.item()`；
2. write gradient 是一次 `grad(loss.sum())`；
3. packed memory mirror 只更新 dirty nodes；
4. topology mutation 后 buffers 与 banks 一致；
5. GPU peak memory 在设定 budget 内。

---

## 24. 最终验收指标

### Router

```text
teacher_student_JS ↓
local_sequence_MI_by_node ↑（仅在 teacher reliable 时）
load_KL bounded
prior_KL 不再爆炸
owner distribution 不再单 branch 单调 collapse
expansion_gain 与实际 NLL improvement 正相关
```

### Controller/write

```text
controller_grad_norm > 0
corr(gW, G_write) > 0
mean selected G_write > C_mem
negative-gain writes = 0
writes_per_sequence <= B
queue evidence 不再永久为 0
```

### Memory/Sleep

```text
coarse_replay_scanned > 0（存在 coarse bank 时）
coarse_absorb_count + coarse_demote_count + coarse_retain_count
  = evaluated coarse rows
sleep residual energy 覆盖 all active nodes
split_consumed_queue_mass 可追溯
evicted_before_sleep 受控
```

### Prune

```text
unknown_leaf_count 可见
unexposed leaf prune count = 0
prune_blocked_by_low_exposure 可见
```

### 性能

```text
Wake 不再逐 sequence × event 调用完整 tree forward
Global/Wake 共用 packed prefixes
write/Sleep replay batch 化
动态 append 与 topology mutation 只发生在 boundary
```

---

## 25. 最终结论

当前模型不应通过以下方式继续修补：

- 单纯提高 `lambda_balance`；
- 单纯延长固定 warm-up；
- 调低或调高 hard action threshold；
- 把 coarse owner memory 强制投影到 leaf；
- 仅增加 top-\(B\)；
- 仅增加 Sleep pressure。

正确 construction 是：

\[
\boxed{
\begin{aligned}
&\text{Router: prior-independent teacher + local soft-EM}\\
&\text{Specialization: sequence MI + reliability-gated load}\\
&\text{Controller: independent gates + counterfactual utility}\\
&\text{Write: true residual gain + threshold + top-}B\\
&\text{Memory: normalized quality retrieval + exact rebase}\\
&\text{Sleep: all-node absorb/demote/retain/split}\\
&\text{Prune: exposure-corrected conditional mass}\\
&\text{Execution: packed GPU evaluate + boundary commit}
\end{aligned}
}
\]

这套 construction 保留当前 actual-frontier 和 bounded subregion 的优势，同时
关闭 Router、Controller/write 和 Memory/Sleep 三个主要反馈缺口。
