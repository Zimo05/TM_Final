# Sleep 阶段 Split / Merge / Promotion / Prune 协调修复记录

## 1. 文档目的

本文记录 Memory Tree 的 sleep 阶段在整合 `Split`、`Merge`、`Promotion` 和 `Prune` 时发现的主要问题、问题产生的原因、修复时采用的算法逻辑，以及修复后建立的关键不变量。

这次修改的目标并不只是让几个函数能够分别运行，而是让它们共享同一套：

- Hawkes replay 语义；
- episodic memory 数据源；
- raw parameter residual 定义；
- 动态树结构状态；
- sleep-cycle 执行顺序；
- checkpoint 恢复逻辑。

涉及的主要文件如下：

- `MemoryResiduals/Replay.py`
- `MemoryResiduals/MemoryBank.py`
- `MemoryResiduals/EpisodicMemory.py`
- `Wake/SequentialController.py`
- `LatentHawkesTree.py`
- `Sleep/Split.py`
- `Sleep/Merge.py`
- `Sleep/Prune.py`
- `Sleep/Coordinator.py`
- `tests/test_sleep_coordination.py`

---

## 2. 总体审查结论

最初的 Split、Merge 和 Prune 各自包含了合理的局部公式，但还没有构成一个严格的结构学习闭环。核心问题主要集中在以下几类：

1. replay window 没有携带正确的 Hawkes 历史；
2. Split 只有优化结果，没有真正写回树；
3. wake 和 sleep 使用了两套不同的 memory 容器；
4. memory usage 没有乘 routing responsibility；
5. Merge、Split 和 Prune 没有统一的执行顺序与冲突仲裁；
6. 新 memory 可能在第一次 sleep 时立刻被删除；
7. leaf effective-mass pruning 只有候选，没有 topology commit；
8. Promotion 使用原始候选数量，且存在 candidate 在自身 window 上自评的问题；
9. 动态 topology 和 sleep patience 无法完整 checkpoint；
10. capacity pruning 与 sleep pruning 使用了不同的 retention 语义。

这些问题中，有些不会导致立即报错，但会让统计量不再代表论文中的数学含义。例如，一个 routing responsibility 几乎为零的 branch 仍然会积累完整 usage，最终使 stale-memory pruning 失效。

---

## 3. Replay window 的 Hawkes 历史错误

### 3.1 原问题

原来的 residual write 只保存：

```python
times[k:end]
types[k:end]
```

但这些时间仍然是原序列中的绝对时间，而且 window 之前的事件历史没有被保存。

随后 Split、Merge 和 Promotion 会把这个局部 window 当成一条从时间 `0` 开始的新 Hawkes 序列。这会产生两个错误：

1. 忽略 window 之前事件留下的 excitation；
2. 错误地把 `[0, times[k]]` 当成一段没有事件的观察区间。

因此，同一个相对事件模式仅仅整体向后平移，就会得到明显不同的 likelihood。修复前的实测中，相同模式平移到时间 `100` 后，event-normalized log-likelihood 相差约 `35.9`。

### 3.2 解决思路

residual 在 wake 阶段是通过下面的目标写出的：

$$
\sum_{j=k}^{e-1}\mathcal L_j(\theta; H_{t_j}).
$$

因此 sleep replay 必须使用相同的 prefix history，并且只累计 `[k,e)` 内的目标事件损失。

### 3.3 实现方案

`EventWindow` 新增：

- `has_full_history`：说明 `times/types` 是否包含完整 prefix；
- `T`：预留观察终点；
- `start_idx/end_idx`：明确目标事件区间。

新写入的 memory 保存 `times[:end]` 和 `types[:end]`，而不是 sliced window。

新增统一入口：

```python
MemoryResiduals.Replay.replay_log_likelihood(...)
```

该函数：

1. 使用完整 prefix 构造事件历史；
2. 只对 `start_idx:end_idx` 求和；
3. 默认按目标事件数量归一化；
4. 对旧 checkpoint 中没有完整历史的 window，先把时间平移到局部零点，再采用明确的 truncated-history 近似。

### 3.4 修复效果

- Split、Merge、Promotion 使用同一个 replay likelihood；
- 同一相对事件模式对整体时间平移保持不变；
- sleep replay 与 wake residual write 的事件范围一致；
- 不再引入虚假的长时间无事件区间。

---

## 4. Split 优化结果没有真正写回树

### 4.1 原问题

`SplitModule` 会优化并输出：

- `theta_plus`；
- `theta_cand`；
- `delta_child`；
- trained split router；
- `should_split`。

但是原来的代码只返回这些结果，没有真正的 split commit。

如果直接调用 `HawkesTree.split_leaf()`：

- child embedding 会随机初始化；
- tree router 会初始化为 50/50；
- `theta_plus` 不会写入 parent；
- `theta_cand` 不会写入 children；
- SplitModule 训练出的 router 会被丢弃；
- 原 leaf memory 不会按 cluster 迁移。

因此 SplitModule 虽然完成了优化，但最终结构与优化结果无关。

### 4.2 参数化上的困难

当前 semantic parameter 来自非线性 hypernetwork：

$$
\theta_n^{\mathrm{sem}}=h_\psi(e_n).
$$

给定目标 `theta_cand`，通常无法精确反求一个 embedding，使：

$$
h_\psi(e_{child})=\theta_{cand}.
$$

如果只随机初始化 child embedding，无法保证 split commit 前后的语义一致性。

### 4.3 解决方案：path-additive semantic offset

在每个 node 上增加 raw Hawkes parameter offset：

$$
\theta_n^{\mathrm{sem}}
=h_\psi(e_n)
+\sum_{v\in path(n)}o_v.
$$

其中 `o_v` 存放在 `tree.semantic_offset` 中。

这样既保留了原本的 path-additive embedding 与共享 hypernetwork，又能通过设置当前 node 的 local offset 精确实现任意 raw parameter target。

新增：

```python
tree.semantic_theta(node_id)
tree.set_semantic_theta(node_id, target_theta)
```

### 4.4 真正的 commit_split

在 `Sleep/Split.py` 中实现 `commit_split(...)`，执行以下步骤：

1. 保存 split 前 parent semantic parameter；
2. 创建两个 children；
3. 将 parent 精确设置为 `theta_plus`；
4. 将两个 children 精确设置为 `theta_cand[0/1]`；
5. 把 SplitModule 的两类 router 转成 tree 使用的 binary logit router；
6. 原 leaf bank 先由 split 后的 internal parent 继承；
7. 用 normalized replay log-likelihood 比较 parent 与两个 children；
8. parent 仅保留其相对最佳 child 的优势超过 hard threshold 的 memory，
   其余 memory 移到解释最好的 child；
9. 对 parent 保留项和 child 移动项分别做 residual rebase；
10. 更新 optimizer、mass EMA、low-mass streak 和 split patience。

### 4.5 Split memory 筛选与 rebase 不变量

对带 replay window 的 memory，定义：

$$
a_r=\log p(w_r\mid\theta_{parent})
-\max_j\log p(w_r\mid\theta_{child,j}).
$$

仅当 $a_r$ 严格大于 `split_memory_hard_threshold` 时保留在 parent；
否则迁移到取得最大 likelihood 的 child。没有 replay window 或 score 非有限的
旧 memory 无法可靠比较，因此保守留在 parent。

假设 memory 在 split 前满足：

$$
\theta_{old}^{\mathrm{sem}}+\Delta\theta_{old}.
$$

无论保留到新 parent 还是移动到 child，都相对目标 node 使用：

$$
\Delta\theta_{new}
=\theta_{old}^{\mathrm{sem}}
+\Delta\theta_{old}
-\theta_{target}^{\mathrm{sem}}.
$$

因此严格保证：

$$
\theta_{target}^{\mathrm{sem}}+\Delta\theta_{new}
=\theta_{old}^{\mathrm{sem}}+\Delta\theta_{old}.
$$

---

## 5. Wake 与 Sleep 使用了两套 Memory

### 5.1 原问题

原来的 `SequentialController` 自己维护：

```python
self.memory_banks
self.split_queues
```

而 Merge 和 Prune 操作的是：

```python
tree.episodic_memory.banks
```

这会导致：

- Split 可能看到 Controller list 中的 memory；
- Merge/Prune 却看到空的 TreeEpisodicMemory；
- topology 改变后，Controller 中仍可能保留已删除 node 的旧 MemoryItem；
- checkpoint 只保存其中一部分状态。

### 5.2 解决方案

`TreeEpisodicMemory` 现在是唯一 memory source of truth。

具体修改：

- Controller novelty 直接查询 `TreeEpisodicMemory.get_bank()`；
- Controller 写 memory 时直接调用 `episodic_memory.add_memory()`；
- 删除 Controller 的第二份 memory list；
- `split_queues` 只保留结构触发计数，不再复制 MemoryItem；
- Split 可以直接从 `MemoryBank` 构造 `SplitBatch`；
- topology commit 后由 coordinator 清理失效 queue。

### 5.3 修复效果

Wake、Split、Merge、Promotion、Prune 和 checkpoint 现在看到的是同一份：

- key；
- residual；
- EventWindow；
- usage；
- cycle usage；
- stale cycles；
- age。

---

## 6. Memory usage 没有乘 routing responsibility

### 6.1 原问题

为了计算所有 leaf 的 episodic correction，模型会读取所有 root-to-leaf path 上的 node bank。

原实现只要某个 bank 被读取，就执行：

$$
\nu_r\leftarrow \nu_r+\alpha_r.
$$

但是“被计算”不等于“对最终预测有效”。如果一个 leaf 的 routing responsibility 接近 0，它的 memory correction 最终也几乎不会进入有效 Hawkes parameter。

修复前实测：

```text
routing = [1.0, 2e-9]
left usage  = 1.0
right usage = 1.0
```

这会让低质量 branch 的 memory 永远看起来被频繁使用，从而无法被 prune。

### 6.2 解决方案

对 node `n`，定义其 routed path mass：

$$
m_n=\sum_{\ell:n\in path(\ell)}r_\ell.
$$

usage 更新改为：

$$
\nu_r\leftarrow \nu_r+\alpha_r m_n.
$$

对于 root，`m_root=1`；对于 leaf-local bank，`m_n` 就是对应 leaf responsibility。

### 6.3 修复效果

相同测试现在得到：

```text
left usage  = 1.0
right usage = 2e-9
```

usage 现在表示 memory 对最终模型的有效贡献，而不是仅仅表示该 bank 在计算图中被访问过。

---

## 7. Memory Pruning 的周期状态

### 7.1 Usage EMA

wake 阶段累计：

$$
\text{cycle\_usage}_r
\leftarrow
\text{cycle\_usage}_r+\alpha_r m_n.
$$

sleep 阶段 consolidation：

$$
\nu_r^{(s+1)}
=\rho_\nu\nu_r^{(s)}
+\text{cycle\_usage}_r.
$$

然后清空 `cycle_usage`。

### 7.2 Staleness

如果当前周期被有效使用：

$$
\eta_r\leftarrow 0.
$$

否则：

$$
\eta_r\leftarrow \eta_r+1.
$$

`stale_cycles` 与普通的事件时间 `age` 分开保存，避免混淆“创建多久”和“多久没有被有效使用”。

### 7.3 Log-domain importance

使用：

$$
\log\xi_r
=\log(\nu_r+\epsilon)-\kappa\eta_r.
$$

删除条件：

$$
\log\xi_r<\log\tau_{prune}.
$$

### 7.4 Fresh-memory grace

原实现中新 memory 的 usage 初始为 0。如果它在刚创建后的第一个 wake/sleep 周期尚未被 retrieve，会立刻得到极低 importance，并可能直接被删除。

现在增加 `min_stale_cycles`，默认值为 2。memory 在至少经历两个没有有效使用的 sleep cycle 后才允许参与删除。

这提供了一个最小试用窗口，同时没有人为增加永久 usage prior。

### 7.5 Capacity pruning 对齐

MemoryBank 达到硬容量时，原来使用：

$$
\frac{1+usage}{1+age}.
$$

这与 sleep 的 usage/staleness 定义不一致。

现在 capacity score 同样考虑：

- usage EMA；
- 当前 cycle usage；
- stale cycles；
- age；
- fresh-memory bonus。

因此 hard capacity 与 sleep pruning 不会再使用完全相反的保留标准。

---

## 8. Merge 与 residual rebase

### 8.1 Merge score

对 sibling leaves `a,b` 和 parent `p`：

$$
\Gamma_{a,b}^{merge}
=\frac{1}{N}
\sum_r
\left[
\max\{\bar L_a(w_r),\bar L_b(w_r)\}
-\bar L_p(w_r)
\right].
$$

其中 replay 使用两个 child bank 的拼接，对应数学上的 memory union，而不是参考代码中无意义的 list subtraction。

### 8.2 Embedding 条件

在 path-additive embedding 下：

$$
e_a=e_p+u_a,\qquad e_b=e_p+u_b,
$$

所以：

$$
\|e_a-e_b\|_2=\|u_a-u_b\|_2.
$$

实现直接比较两个 sibling-specific node embedding。

### 8.3 Merge residual rebase

child 被删除、parent 成为 leaf 后，memory residual 必须从 child semantic baseline 转换到 parent baseline：

$$
\Delta\theta_{r\rightarrow p}
=\theta_c^{sem}
+\Delta\theta_r
-\theta_p^{sem}.
$$

严格保证：

$$
\theta_p^{sem}+\Delta\theta_{r\rightarrow p}
=\theta_c^{sem}+\Delta\theta_r.
$$

Merge commit 同时迁移：

- keys；
- rebased deltas；
- usage；
- cycle usage；
- stale cycles；
- age；
- EventWindow origin。

迁移顺序是先将 parent、child `a`、child `b` 三个 bank 完整合并到
parent bank，再统一执行 sequence 筛选；parent semantic parameter 在整个
commit 中保持不变。对合并后的 memory 定义：

$$
g_r=
\bar L( w_r\mid\theta_p^{sem}+\Delta\theta_{r\rightarrow p})
-\bar L(w_r\mid\theta_p^{sem}).
$$

仅保留 $g_r\leq$ `merge_memory_hard_threshold` 的 sequence。这表示 parent
参数相对 memory-specific effective parameter 的解释损失不能超过 hard
threshold。没有 replay window 的 legacy item 无法评分，因此保守保留。

筛选完成后才执行 bank capacity pruning。

并移除：

- child topology nodes；
- child embeddings；
- child semantic offsets；
- parent router；
- optimizer 中失效参数；
- child pruning state。

---

## 9. Promotion 作为 Merge 的证据

### 9.1 Soft support

candidate `r` 对 branch `c` 的 support：

$$
S_{r,c}=\sum_{j\in\mathcal M_c}
\sigma\left(
\frac{sim(k_r,k_j)-\tau_{sim}}{T_{sim}}
\right).
$$

### 9.2 Utility gain

$$
G_{r,c}
=
\frac{
\sum_j\omega_{rj}
\left[
\bar L(w_j\mid\theta_c+\Delta\theta_r)
-\bar L(w_j\mid\theta_c)
\right]
}{S_{r,c}+\epsilon}.
$$

### 9.3 Balance

使用正确公式：

$$
B_r
=\frac{2\min(S_{r,a},S_{r,b})}
{S_{r,a}+S_{r,b}+\epsilon}.
$$

没有采用参考片段中误写的 `2.0 - ...`。

### 9.4 发现的统计偏差

原 promotion 设计有两个问题：

1. candidate 会在包含自身 window 的 source bank 上评估 gain，存在自评偏差；
2. merge 使用 `promotion_count`，bank 越大越容易通过，统计量依赖 capacity。

### 9.5 修复方案

- source branch 使用 leave-one-out，排除 candidate 自己的 replay window；
- merge 同时要求：
  - `promotion_count >= min_promotions`；
  - `promotion_ratio >= min_promotion_ratio`。

其中：

$$
promotion\_ratio
=\frac{\#\text{accepted candidates}}
{\#\text{valid replay memories}}.
$$

### 9.6 Promotion 为什么不做 rebase

Promotion 时 children 并没有被删除。parent 位于两个 child path 上，因此把 residual 从 source child 移到 parent 后：

- source branch 仍然会经过 parent；
- residual raw correction 保持不变；
- sibling branch 获得同一个共享 correction。

因此 Promotion 与 Merge 不同，不需要 semantic baseline rebase。只有 topology compression 删除 children 时才执行 merge rebase。

---

## 10. Leaf Effective-Mass Pruning

### 10.1 Mass EMA

当前周期 leaf mass：

$$
m_\ell^{(s)}
=\frac{1}{|B_s|}
\sum_{i\in B_s}r_{i\ell}.
$$

EMA：

$$
\bar m_\ell^{(s)}
=\beta_m\bar m_\ell^{(s-1)}
+(1-\beta_m)m_\ell^{(s)}.
$$

只有连续 `patience` 个 sleep cycle 满足：

$$
\bar m_\ell<\tau_{mass}
$$

才生成 low-mass candidate。

### 10.2 原问题

原来只有 candidate collection，没有 topology commit。直接删除一个 leaf 会破坏严格二叉树，因为 parent 会只剩一个 child，router/path mask 都会失效。

### 10.3 commit_leaf_prune

实现了 leaf-sibling 情况下的精确 collapse：

1. 删除低质量 leaf；
2. 选择 surviving leaf sibling；
3. 将 parent 变成 leaf；
4. 将 parent semantic 设置为 surviving sibling semantic；
5. 在 topology 变化前合并 parent、survivor 和 removed leaf 三个 bank 的
   memory candidate；
6. 使用 survivor/new-parent semantic parameter 统一筛选 replay sequence；
7. 所有保留 memory 根据各自 source semantic baseline rebase 到新 parent；
8. 删除两个旧 child nodes、child banks 和 parent router；
9. 更新 optimizer、mass EMA、streak 和 structure buffers；
10. 最后执行 capacity pruning 和全局 stale-memory pruning。

对来源 node $x$ 的 memory，筛选量为：

$$
g_r=
\bar L(w_r\mid\theta_x^{sem}+\Delta\theta_r)
-\bar L(w_r\mid\theta_s^{sem}).
$$

仅保留 $g_r\leq$ `prune_memory_hard_threshold` 的 sequence，其中
$\theta_s^{sem}$ 是 survivor、也是 prune 后的新 parent parameter。来自旧
parent 或 survivor 且没有 replay window 的 legacy memory 保守保留；来自
removed leaf 且无法评分的 memory 直接删除。

保留项使用：

$$
\Delta\theta_r^{new}
=\theta_x^{sem}+\Delta\theta_r-\theta_s^{sem},
$$

从而严格保持 source memory 的 effective parameter 不变。

当前实现明确只处理 sibling 也是 leaf 的情况。如果 sibling 是内部子树，则跳过 candidate，因为那需要单独的 subtree promotion/collapse 算法，不能用 leaf-sibling 逻辑强行处理。

---

## 11. 统一 Sleep Transaction

### 11.1 原问题

如果没有统一顺序，会出现以下冲突：

- Prune 先执行：删掉 Split/Merge 所需 replay evidence；
- topology commit 先执行：旧 responsibility 的 leaf 维度失效；
- 同一个 leaf 同时被 Split 和 Merge 选中；
- 先 Promotion：child bank 变小，随后 Merge 可能变成 insufficient replay；
- Merge 后外部 split queue 仍指向已经删除的 leaf。

### 11.2 新增 Coordinator

新增：

```python
Sleep.Coordinator.run_sleep_cycle(...)
```

它把一次 sleep 当成结构事务处理。

### 11.3 固定执行顺序

1. 冻结 pre-change leaf/topology snapshot；
2. consolidate cycle usage，但暂不删除 memory；
3. 使用旧 topology 的 responsibility 更新 leaf mass；
4. 在同一份未裁剪 replay snapshot 上评估所有 sibling merge；
5. 按优先级提交结构动作：
   - Merge；
   - low-mass leaf collapse；
   - Split；
6. 仅对未发生结构变化的 sibling pair 执行 Promotion；
7. 最后执行 stale-memory pruning；
8. 清理失效 split queue 和动态状态。

### 11.4 冲突原则

#### Merge 优先于 Split

Merge 同时使用：

- replay likelihood；
- sibling embedding distance；
- cross-branch promotion evidence。

如果同一对 leaves 已经满足 compression 条件，再对其中一个 leaf split 会产生相互矛盾的结构动作，因此 Merge 优先。

#### Low-mass collapse 优先于 Split

如果 leaf 已连续多个周期质量过低，应先删除无效 branch，而不是继续增加其结构复杂度。

#### 两个 sibling 都低质量时不任意选择 survivor

如果两个 siblings 都是 low-mass candidate，不能依赖遍历顺序任意保留一个。此时只有在 Merge 判据通过时才压缩，否则留到后续周期处理。

#### Promotion 只作用于结构未变化的 pair

如果 pair 已 merge，则 Promotion 是冗余的；如果 leaf 已 split/prune，则旧 candidate index 已失效。因此 Promotion 放在结构 commit 之后，并只处理 untouched pairs。

---

## 12. 动态 Topology 与 Checkpoint

### 12.1 原问题

普通 `state_dict` 能看到动态添加的 node parameters，但新建模型仍保持初始 topology。直接加载动态 split 后的 state 会出现：

```text
Unexpected key(s): node_emb.root_L_L, routers.root_L...
```

同时以下状态也没有保存：

- `mass_ema`；
- `low_mass_streak`；
- Split 的 consecutive-ready patience；
- 动态 router 类型。

### 12.2 解决方案

`HawkesTree.get_extra_state()` 保存：

- topology parent/left/right/depth；
- 动态 router modules；
- mass EMA；
- low-mass streak。

在 `load_state_dict` 的 pre-hook 中，先根据 extra state 重建：

- `nodes`；
- `node_emb`；
- `semantic_offset`；
- `routers`；
- routing/path buffers。

然后再加载 tensor parameters。

`SplitModule` 也通过 extra state 保存 `consecutive_ready`。

### 12.3 修复效果

模型在动态 Split/Merge 后可以加载到初始 topology 不同的新实例，并恢复：

- leaf IDs；
- internal IDs；
- semantic parameters；
- router outputs；
- mass EMA；
- low-mass patience；
- Split commit patience。

---

## 13. 关键不变量汇总

### 13.1 Memory rebase 不变量

所有 topology relocation 都应满足：

$$
\theta_{new}^{sem}+\Delta_{new}
=\theta_{old}^{sem}+\Delta_{old}.
$$

已用于：

- Split parent-to-child memory relocation；
- Merge child-to-parent relocation；
- leaf collapse survivor-to-parent relocation；
- leaf collapse 时 parent shared memory 的 baseline 更新。

### 13.2 Replay snapshot 不变量

同一个 sleep cycle 中：

- Split、Merge 和 Promotion 必须基于同一份 pre-prune replay；
- 结构判断阶段不能偷偷修改 memory bank；
- topology commit 前后的 node ID/index 不能混用。

### 13.3 Usage 不变量

usage 表示 memory 对最终有效参数的贡献，而不是表示该 bank 被程序计算过：

$$
\Delta usage_r=\alpha_r\times routed\ path\ mass.
$$

### 13.4 二叉拓扑不变量

任何 commit 后，每个 node 必须满足：

- leaf：`left=None` 且 `right=None`；
- internal：左右 children 同时存在；
- router 只属于 internal node；
- routing masks 与当前 topology 一致。

---

## 14. 回归测试

新增：

```text
tests/test_sleep_coordination.py
```

标准运行方式：

```bash
/Users/zimoshen/opt/miniconda3/envs/memTree/bin/python -m unittest discover -v
```

当前测试覆盖：

1. replay likelihood 对整体时间平移保持不变；
2. replay 正确使用 prefix history；
3. all-path usage 乘 routing responsibility；
4. Split commit 前后 memory effective parameter 不变；
5. Merge commit 前后 memory effective parameter 不变；
6. fresh-memory grace；
7. Merge 与 Split 冲突时 Merge 优先；
8. 动态 topology checkpoint；
9. mass EMA checkpoint。

当前结果：

```text
Ran 19 tests
OK
```

---

## 15. 当前明确边界

目前结构操作严格支持：

- leaf split；
- leaf-sibling merge；
- leaf-sibling promotion；
- leaf-sibling low-mass collapse。

暂不自动处理：

- leaf 与内部子树相邻时的 subtree promotion；
- 任意深度 subtree merge；
- 两个低质量 sibling 在 merge 证据不足时的强制删除；
- 跨 branch 的全局最优结构搜索。

这些情况会被安全地跳过，而不是破坏二叉 topology。

---

## 16. 后续维护建议

1. 结构性 sleep cycle 统一调用 `run_sleep_cycle()`，不要手动随意排列 Merge、Split 和 Prune。
2. 直接调用 `sleep_phase_prune()` 只适用于本周期没有 topology change 的场景。
3. 新增结构操作时，必须写出 residual relocation 的 old/new semantic baseline，并验证 effective parameter 不变量。
4. 新增 replay 统计时，应明确：
   - 是否使用完整历史；
   - 是否 event-normalized；
   - 是否使用独立/leave-one-out evidence；
   - 是否在判断过程中修改状态。
5. 新增动态 node state 时，应同步处理：
   - `.to(device/dtype)`；
   - checkpoint；
   - Split；
   - Merge；
   - leaf collapse；
   - capacity prune。
6. Promotion、Merge 等阈值应在验证集上按比例或有效质量校准，避免使用随 bank capacity 变化的绝对计数。

---

# Train 与 Inference 算法逻辑记录

## 17. Train 文件夹的职责划分

在完成 Wake、Sleep、MemoryBank 和动态 topology 的统一后，`Train` 文件夹负责把这些局部算法组织成论文中的完整训练和推理流程。

当前文件职责如下：

### `Train/ConstructTree.py`

负责：

- 从 CSV 读取事件序列；
- 检查时间与事件类型；
- 将原始 event type 映射到 `[0,D-1]`；
- 构造 `HawkesFamily`；
- 执行可选的 Hawkes cold start；
- 构造初始 `HawkesTree`、episodic memory 和 working memory。

### `Train/Train.py`

负责：

- causal prefix encoding；
- wake objective；
- working-memory 在线更新；
- hard action 与 memory write；
- sleep replay objective；
- Hawkes stability regularization；
- Split proposal 优化；
- 调用统一 Sleep Coordinator；
- epoch/full-objective 统计；
- 动态树 checkpoint。

### `Train/Inference.py`

负责：

- 从 checkpoint 重建动态 topology；
- 只运行 wake mechanism；
- 在线 routing 和 episodic retrieval；
- working-memory fast adaptation；
- 无未来泄漏的 delayed memory write；
- observed-event prediction diagnostics；
- next-event type/time forecast。

### `Train/README.md`

记录训练、推理 API、CLI 和目标函数的对应关系。

### `Train/__init__.py`

保持轻量，只声明 package。具体 API 从 `Train.Train` 和 `Train.Inference` 导入，避免执行 `python -m Train.Train` 时发生模块重复加载。

---

## 18. ConstructTree 的数据和冷启动逻辑

### 18.1 输入数据格式

CSV 至少需要两列：

```text
event_times
event_types
```

每行代表一条事件序列。读取后转换为：

```python
{
    "times": FloatTensor[K],
    "types": LongTensor[K],
    "T": scalar_tensor,
}
```

其中要求：

- `times` 和 `types` 长度相同；
- 时间非负；
- 时间按非递减顺序排列；
- event type 被映射为连续整数。

### 18.2 Hawkes cold start

Cold start 在构建 memory tree 之前，用整个数据集训练一个普通多元 Hawkes 模型。

它优化：

$$
\mathcal L_{cold}
=\frac{1}{N_{event}}\sum_i
\operatorname{NLL}(w_i)
+\lambda_{stab}\mathcal R_{stab}
+\lambda_{L1}\|W\|_1.
$$

作用是：

- 给出合理的 decay/intensity 时间尺度；
- 在 memory tree 学习前确认 Hawkes likelihood 数值稳定；
- 提供一个可独立检查的 backbone checkpoint。

原文件曾引用不存在的 `HawkesParameters`。现在统一使用仓库真实实现 `HawkesFamily`。

### 18.3 Tree 初始化

构造的 `HawkesTree` 包含：

- differentiable routers；
- path-additive node embeddings；
- raw semantic offsets；
- Hawkes hypernetwork；
- `TreeEpisodicMemory`；
- `WorkingMemoryAdapter`。

Cold-start backbone 的 raw parameter 不直接替代 tree semantic parameter。后续训练由 wake/sleep replay 继续学习 tree hypernetwork 和 semantic offsets。

---

## 19. Causal Prefix Encoder

### 19.1 为什么必须严格 causal

预测第 `k` 个事件时，encoder 只能观察：

$$
\{(t_j,c_j):j<k\}.
$$

不能把当前事件或未来事件传入 latent context，否则 routing 和 prediction 都会发生 label leakage。

### 19.2 输入特征

`CausalPrefixEncoder` 对每个已观察事件构造：

- event type embedding；
- `log(1+t_j)`；
- `log(1+\Delta t_j)`。

两类时间特征分别表示：

- 全局序列位置；
- 局部 inter-arrival dynamics。

拼接后的特征送入 GRU：

$$
h_j=\operatorname{GRU}(x_j,h_{j-1}),
$$

再投影为 tree context：

$$
z_k=\tanh(W_zh_{k-1}+b_z).
$$

对于空 prefix，即 `k=0`，使用可训练的 `empty_prefix`。

### 19.3 训练中的实现选择

训练时，每个 event 都从严格 prefix 重新计算 encoder output，而不是复用已经经历 optimizer 更新的旧 hidden state。

这样虽然计算量更高，但能避免：

- 跨 optimizer step 持有失效 autograd graph；
- hidden state 对应旧 encoder parameter；
- 不明确的 truncated BPTT 语义。

---

## 20. Wake Objective

论文中的 wake objective 为：

$$
\mathcal L_{wake}
=\sum_{i=1}^{N}\sum_{k=1}^{K_i}
\ell_k^{pred}(\tilde\theta_{t_k}^{eff})
+\lambda_{wm}\sum_t\|\Delta\tilde\theta_t^{wm}\|_2^2
+\lambda_{write}\sum_t
\mathbf 1[a_t=\mathrm{MEMORIZE}].
$$

代码按事件进行 stochastic optimization：

$$
\mathcal L_{wake,t}
=\ell_t^{pred}
+\lambda_{wm}\|\Delta\theta_t^{wm}\|_2^2
+\lambda_{write}\mathbf 1[a_t=\mathrm{MEMORIZE}].
$$

### 20.1 每个事件的计算顺序

预测 event `k` 时执行：

1. encoder 读取 `events[:k]` 得到 `z_k`；
2. tree 根据 `z_k` 计算 leaf responsibility；
3. 所有相关 node bank 根据 memory query 执行 sparse retrieval；
4. episodic usage 乘 routed path mass；
5. 组合 semantic、episodic 和 working residual；
6. 计算当前事件 Hawkes prediction NLL；
7. 根据 surprise、novelty 和 memory count 决定 action；
8. 计算 wake objective；
9. 更新 tree、router、retriever、query net 和 encoder；
10. 使用 working gradient 更新 fast state；
11. 推进 memory age；
12. 如需要则写入 episodic memory。

### 20.2 Effective Hawkes parameter

对 leaf responsibility `r_{t\ell}`，最终 raw parameter 为：

$$
\theta_t^{eff}
=\sum_{\ell}r_{t\ell}
\left(
\theta_\ell^{sem}
+\Delta\theta_{t\ell}^{epi}
\right)
+\Delta\theta_t^{wm}.
$$

然后再通过 softplus 得到正的：

- baseline intensity `mu`；
- excitation weights `W`。

### 20.3 Prediction loss

当前事件损失为：

$$
\ell_k^{pred}
=-\log\lambda_{c_k}(t_k\mid H_{t_k})
+\sum_d\int_{t_{k-1}}^{t_k}
\lambda_d(\tau\mid H_\tau)d\tau.
$$

该损失通过 effective parameters 对以下模块产生梯度：

- semantic hypernetwork；
- semantic offsets；
- tree routers；
- memory query network；
- sparse retriever；
- causal prefix encoder。

---

## 21. Working Memory 的在线更新

### 21.1 Working residual

每个 sequence 开始时：

```python
tree.reset_working_memory()
```

当前事件使用一个由 persistent fast state clone 出来的可微 tensor：

```python
working_delta = working_memory.make_trainable_delta()
```

### 21.2 正则化

Wake objective 加入：

$$
\lambda_{wm}\|\Delta\theta_t^{wm}\|_2^2.
$$

作用是防止 working memory 用过大的瞬时 residual 覆盖 semantic/episodic knowledge。

### 21.3 更新顺序

先计算：

$$
g_t
=\nabla_{\Delta\theta_t^{wm}}
\left(
\ell_t^{pred}
+\lambda_{wm}\|\Delta\theta_t^{wm}\|_2^2
\right).
$$

为了同时保留主模型的 backward graph，代码使用 `autograd.grad(..., retain_graph=True)` 得到 working gradient，然后：

1. `wake_loss.backward()` 更新慢参数；
2. optimizer step；
3. `working_memory.update_from_gradient(g_t)` 更新 fast state。

Working update 为：

$$
\Delta\theta_{t+1}^{wm}
=\rho_{wm}\Delta\theta_t^{wm}
-\eta_{wm}g_t.
$$

### 21.4 为什么 memory age 必须最后更新

retriever score 在计算图中读取 `bank.age`。如果在 backward 前对 `age` 做 in-place update，会触发 tensor version mismatch。

因此训练和 inference 都在当前事件所有 gradient 完成之后才执行：

```python
episodic_memory.step_age()
```

---

## 22. Wake Action 与 Memory Write

### 22.1 Hard action

Controller 根据：

- surprise；
- novelty；
- 相似 memory 数量；

选择：

- `ASSIMILATE`；
- `RETRIEVE`；
- `MEMORIZE`；
- `QUEUE_SPLIT`。

### 22.2 Write penalty

严格对应论文，`lambda_write` 只惩罚：

```text
action == MEMORIZE
```

`QUEUE_SPLIT` 仍然会保存 residual evidence，但它的结构复杂度由 Split gate 和 sleep tree penalty 约束，不重复计入论文中的 `1[a_t=MEMORIZE]`。

### 22.3 Residual write

对于被选中的 leaf semantic parameter，计算未来局部窗口损失梯度：

$$
\Delta\theta_r
=P_r\left(
-\eta_{mem}
\nabla_{\theta_\ell^{sem}}
\sum_{j=k}^{k+h}
\ell_j(\theta_\ell^{sem})
\right).
$$

其中 `P_r` 是 residual projection，例如 low-rank projection。

训练是 offline 场景，因此可以直接访问完整 future write horizon。写入的 `EventWindow` 保存完整 prefix history，使 sleep replay 与 write 时的 Hawkes history 一致。

---

## 23. Sleep Objective

论文中的 sleep objective 为：

$$
\mathcal L_{sleep}
=-\sum_{w\in B\cup\tilde B}
\log p(w\mid T,\{\mathcal M_n\})
+\lambda_{tree}\mathcal L(T)
+\lambda_{mem}\sum_{n\in\mathcal V(T)}|\mathcal M_n|
+\lambda_{stab}\mathcal R_{stab}.
$$

### 23.1 Replay reconstruction

当前实现遍历所有有效 node bank。对 memory `r`：

$$
\theta_r^{eff}
=\theta_{origin(r)}^{sem}
+\Delta\theta_r.
$$

然后使用统一的 conditional replay likelihood，只计算 EventWindow 的目标事件区间。

不同长度 replay window 默认按事件数量归一化，避免长窗口在 sleep gradient 中获得不成比例的权重。

如果当前没有任何 replay，代码构造一个与 tree parameter 相连的零 replay loss，使 stability term 仍可正常 backward。

### 23.2 Tree complexity

当前：

$$
\mathcal L(T)=|I(T)|,
$$

即 internal node 数量。

这是离散项，本身不会直接对 node count 求梯度。其结构作用通过以下机制实现：

- SplitModule 中的 differentiable gate penalty；
- Coordinator 的 Merge/Split 冲突仲裁；
- low-mass collapse。

### 23.3 Memory count

$$
\sum_n|\mathcal M_n|
$$

同样是离散项。它的实际结构作用通过：

- write penalty；
- capacity pruning；
- sleep stale-memory pruning；

来实现。

### 23.4 Hawkes stability regularizer

对每个 leaf：

$$
A_{dd'}
=\sum_{m=1}^M
\frac{\operatorname{softplus}(\tilde W_{dd'm})}
{\delta_m}.
$$

稳定性惩罚为：

$$
\mathcal R_{stab}
=\max(0,\rho(A)-\tau_{stab})^2.
$$

代码使用 non-negative matrix 的 power iteration 估计 spectral radius，避免对重复 eigenvalue 使用不稳定的 `eigvals` backward。

### 23.5 Beta sleep

Sleep phase 实际 backward：

$$
\beta_{sleep}\mathcal L_{sleep}.
$$

因此 `beta_sleep` 不只是日志中的系数，而是直接缩放 sleep gradient。

---

## 24. Split Proposal 与 Sleep Coordinator

### 24.1 SplitModule 生命周期

Trainer 为当前每个 leaf 维护一个持久的 `SplitModule`。

当 topology 变化时：

- 删除已经不是 leaf 的 module；
- 为新 children 创建 module；
- 保留仍然存在的 leaf module 和 consecutive-ready patience。

### 24.2 Split trigger

默认只有满足以下条件的 leaf 才优化 split proposal：

```text
controller.split_queues[leaf_id] > 0
```

这使 Split 不会仅因为 bank 中存在普通 memories 就持续尝试增长结构。

可通过：

```python
require_split_trigger=False
```

关闭该限制，用于消融或实验。

### 24.3 Split replay batch

Split 直接从唯一的 `TreeEpisodicMemory` bank 构造：

- residuals；
- normalized keys/contexts；
- usage/age weights；
- conditional EventWindows。

### 24.4 Sleep transaction

连续参数完成 sleep optimizer step 后，Trainer 才调用：

```python
run_sleep_cycle(...)
```

结构动作顺序为：

```text
Merge > low-mass leaf collapse > Split > Promotion > Memory Prune
```

所有 Merge/Split 判断使用同一份 pre-prune snapshot。

---

## 25. Full Training Loop

### 25.1 Epoch 流程

每个 epoch：

1. 随机打乱 sequence 顺序；
2. 对每条 sequence 重置 working memory；
3. 逐事件优化 wake objective；
4. 收集当前固定 topology 下的 responsibility；
5. 每隔 `sleep_every` 个 epoch 执行 sleep phase；
6. sleep continuous optimization；
7. Split proposal optimization；
8. Coordinator structural transaction；
9. 保存 checkpoint。

Topology 只在 sleep 周期末修改，因此同一个 wake epoch 中收集的 responsibility 具有一致的 leaf dimension。

### 25.2 Full objective 日志

Trainer 记录：

$$
\mathcal L
=\mathcal L_{wake}
+\beta_{sleep}\mathcal L_{sleep}.
$$

为了使不同事件数量的 epoch 可比较，wake 日志按事件数归一化。Sleep replay 内部也采用 event-normalized window mean。

### 25.3 Gradient clipping

Wake 和 sleep 都执行 gradient clipping，并对非有限 loss/gradient 抛出错误，防止 Hawkes intensity 或 residual 更新产生静默 NaN。

---

## 26. Training Checkpoint

每个 epoch 保存：

- dynamic tree state；
- topology extra state；
- episodic memory banks；
- Hawkes backbone；
- causal encoder；
- optimizer state；
- per-leaf SplitModule states；
- Wake/Sleep/Structure/Training configs；
- history；
- model dimensions；
- memory capacity；
- tree temperature；
- hypernetwork hidden dimension；
- working-memory rho/eta；
- decays。

Inference loader 会先根据 checkpoint config 构建一个最小 tree，然后通过 tree 的 load pre-hook 恢复真实动态 topology，再加载 parameter 和 memory state。

对于自定义 encoder，`MemoryTreeInference.from_checkpoint()` 支持显式传入 encoder 实例，再加载其 state dict。默认情况下自动重建 `CausalPrefixEncoder`。

---

## 27. Wake-Only Inference

论文要求 inference 时只使用 wake mechanism，不执行 sleep objective。

因此 `MemoryTreeInference.run_sequence()` 会执行：

- causal prefix encoding；
- tree routing；
- episodic retrieval；
- responsibility-weighted usage；
- effective Hawkes prediction；
- working-memory online adaptation；
- hard action；
- 可选 memory write。

不会执行：

- sleep replay optimization；
- Split commit；
- Merge commit；
- Promotion；
- topology prune。

所以一条 inference sequence 内的 topology 保持固定。

### 27.1 Observed-event output

对每个已观察事件返回：

- event NLL；
- 当前时刻各事件类型 intensity；
- predicted type；
- true type；
- leaf responsibility；
- selected leaf；
- wake action。

### 27.2 Working-memory adaptation

Inference 不更新慢模型 parameter，但会保留只关于 `working_delta` 的 autograd graph，并执行 fast-state update。

可通过：

```python
InferenceConfig(adapt_working_memory=False)
```

关闭。

### 27.3 Usage update

默认 inference 会更新 memory usage 和 age，因为这属于 wake mechanism 的在线状态。

可通过：

```python
InferenceConfig(update_memory_usage=False)
```

进行完全只读评估。

---

## 28. Inference Memory Write 的未来泄漏控制

### 28.1 问题

Residual write 需要未来 horizon：

$$
\sum_{j=k}^{k+h}\ell_j.
$$

训练时完整 sequence 已知，可以离线直接计算。但在线 inference 在 event `k` 时不能访问 `k+1:k+h`。

### 28.2 Delayed write

当 action 请求 memory write 时，Inference 只保存 pending request：

- origin event index；
- ready index；
- selected leaf；
- query；
- 当时的 semantic snapshot；
- action type。

只有当：

$$
current\_index\ge k+h
$$

时才真正生成 residual 并写入 bank。

序列结束时尚未达到 horizon 的请求保留为 `pending_write_count`，不会用不完整未来强行写入。

这保证 inference 不会因为函数已经拿到了完整 tensor，就在逻辑上偷看尚未发生的事件。

---

## 29. Next-Event Forecast

`predict_next_event()` 从完整 observed prefix 得到当前 effective Hawkes intensity：

$$
\lambda_d(t\mid H_t).
$$

### 29.1 Event type

类型概率：

$$
p(c=d\mid H_t)
=\frac{\lambda_d(t\mid H_t)}
{\sum_j\lambda_j(t\mid H_t)}.
$$

预测类型取最大概率类型。

### 29.2 Event time

当前实现使用 locally constant total rate：

$$
\mathbb E[\Delta t]
\approx
\frac{1}{\sum_d\lambda_d(t\mid H_t)}.
$$

所以：

$$
\hat t_{next}=t+\mathbb E[\Delta t].
$$

这是确定性 forecast，不是精确 Hawkes sampling。若需要严格采样，应另外实现 Ogata thinning；当前接口在文档和返回值中明确这一近似，避免把局部 rate expectation 误称为精确生成。

---

## 30. Train 与 Inference 的使用方式

### 30.1 Python 训练

```python
from Train.Train import CausalPrefixEncoder, MemoryTreeTrainer

encoder = CausalPrefixEncoder(
    num_event_types=D,
    z_dim=tree.z_dim,
)

trainer = MemoryTreeTrainer(
    tree=tree,
    hawkes=hawkes,
    encoder=encoder,
)

history = trainer.train(dataset)
```

### 30.2 CLI 训练

```bash
python -m Train.Train \
  --data-path data.csv \
  --epochs 20 \
  --cold-start-epochs 5 \
  --checkpoint checkpoints/memory_tree.pt \
  --num-basis 2 \
  --decays 0.5 1.5
```

### 30.3 Python 推理

```python
from Train.Inference import MemoryTreeInference

inference = MemoryTreeInference.from_checkpoint(
    "checkpoints/memory_tree.pt"
)

online_result = inference.run_sequence(sequence)
forecast = inference.predict_next_event(sequence)
```

### 30.4 CLI 推理

```bash
python -m Train.Inference \
  --checkpoint checkpoints/memory_tree.pt \
  --times 0.1 0.4 0.9 \
  --types 0 1 0
```

使用 `--no-write` 可以关闭 inference episodic-memory mutation。

---

## 31. Train / Inference 回归测试

新增：

```text
tests/test_train_inference.py
```

覆盖：

1. 微型数据完成 wake training；
2. sleep objective 正常 backward；
3. sleep transaction 正常执行；
4. offline memory write 使用 full-history EventWindow；
5. checkpoint 能恢复动态 tree 和 memories；
6. inference 只运行 wake，不改变 topology；
7. working-memory inference adaptation；
8. next-event type probability 归一化；
9. delayed write 不读取尚未到达的 future horizon；
10. checkpoint 恢复 memory capacity 和模型超参数。

完整测试命令：

```bash
/Users/zimoshen/opt/miniconda3/envs/memTree/bin/python \
  -m unittest discover -v
```

当前结果：

```text
Ran 19 tests
OK
```

---

## 32. 当前训练与推理边界

1. `predict_next_event()` 的时间是 local-rate expectation，不是 Ogata 精确采样。
2. Inference 不执行 topology change；`QUEUE_SPLIT` 只累计 trigger，等待后续 sleep。
3. 自定义 encoder 在 checkpoint load 时需要显式传入相同结构的 encoder 实例。
4. Sleep tree/memory count 是离散项，其结构效果由 Coordinator 和 pruning 落实，而不是直接对 node count 求梯度。
5. 当前训练循环按事件执行 optimizer step，优先保证 streaming/working-memory 语义；若以后改为 sequence-level BPTT，需要重新处理 working-state inner update 和 memory mutation 的计算图边界。
6. Training 与 inference 都必须使用 delayed write；完整 future horizon 只能在对应事件真实到达后用于构造 residual。

---

# End-to-End 关键断点修复记录（2026-07-13）

## 33. 修复背景与审计结论

本次工作从真实训练入口 `Train/Train.py` 出发，对下面这条链进行了静态审计、实际 NLL 梯度检查和微型端到端训练：

$$
\text{strict prefix encoder}
\rightarrow r_{t\ell}
\rightarrow \Delta\tilde\theta^{epi}_{t,\ell}
\rightarrow \tilde\theta_t^{eff}
\rightarrow \ell_t^{pred}.
$$

审计确认 semantic parameter、routing、working memory、episodic retrieval、sleep transaction 和动态 split 已经存在，但发现四个会直接影响训练正确性的断点：

1. wake training 会立即写入由 future horizon 构造的 residual；
2. Hawkes cold start 参数没有初始化 semantic tree；
3. entmax 单 support 会令 query/retriever 梯度永久为零；
4. sleep replay 绕过 encoder、routing 和 retrieval，只更新 semantic 参数。

下面按“问题 → 修改方法 → 修改结果”记录每一项修复。

---

## 34. Training Residual Write 的未来信息泄漏

### 34.1 问题

written residual 的定义使用事件窗口：

$$
\Delta\tilde\theta_t^{write}
=P_r\left(
-\eta_m\nabla_{\tilde\theta}
\sum_{\tau=t}^{t+h}\ell_\tau^{pred}
\right).
$$

旧训练循环在 event `k` 完成后，直接用完整 sequence 中的 `k:k+h` 生成 residual 并写入 bank。这样在预测 `k+1` 时，memory 可能已经包含一个使用过 `k+1` 甚至更远事件标签的 residual。

即使 prefix encoder 严格只读取 `events[:k]`，这一 memory mutation 仍然构成 look-ahead leakage。

### 34.2 修改方法

训练侧改为与 inference 相同的 delayed-write transaction：

1. action 在 event `k` 触发 write 时，只创建 pending request；
2. request 保存 `event_index`、`ready_index=k+h`、leaf、query、当时的 semantic snapshot 和 action；
3. 当前事件完成预测、backward 和 optimizer step 后，才检查 ready requests；
4. 只有当：

   $$
   current\_index\ge k+h
   $$

   时才调用 `write_residual_memory()`；
5. `QUEUE_SPLIT` counter 也只在 residual 真正 commit 后增加；
6. 序列结束时尚未 ready 的请求保留为 `pending_write_count`，不会用不完整 horizon 强制写入。

实现集中在：

```text
Train/Train.py
  _make_write_request()
  _commit_write_request()
  train_wake_sequence()
```

同时更新 `Wake/SequentialController.py` 的接口说明，明确调用方必须等待完整 horizon 到达。

### 34.3 修改结果

- 训练和 inference 现在具有一致的因果 write 语义；
- residual 不可能参与预测其自身构造窗口内的事件；
- `write_horizon=2`、sequence 长度为 3 时，只会 commit event `0` 的一条 write，event `1/2` 的请求保持 pending；
- delayed `QUEUE_SPLIT` 仍能积累 trigger 并在 sleep 阶段成功 split。

---

## 35. Hawkes Cold Start 与 Semantic Tree 断链

### 35.1 问题

standalone Hawkes cold start 优化：

```text
hawkes.raw_mu
hawkes.raw_W
```

但 Memory wake/sleep 的 prediction NLL 使用 tree 生成的：

$$
\tilde\theta_t^{eff}
=\sum_\ell r_{t\ell}
\left(\tilde\theta_\ell^{sem}+\Delta\tilde\theta_{t,\ell}^{epi}\right)
+\Delta\tilde\theta_t^{wm}.
$$

`MemoryTreeTrainer` 的 optimizer 只包含 tree 和 encoder。Hawkes 对象在此阶段主要作为 likelihood implementation 使用，因此 cold-start 后的 `raw_mu/raw_W` 原本不会进入 semantic forward。

结果是 `--cold-start-epochs` 虽然消耗了训练时间并保存 checkpoint，却没有给 Memory Tree 提供实际初始化。

### 35.2 修改方法

在 `LatentHawkesTree.py` 新增：

```python
HawkesTree.initialize_semantics_from_hawkes(hawkes)
```

该方法构造 cold-start raw target：

$$
\tilde\theta^{cold}
=\operatorname{concat}(raw\_mu,\operatorname{vec}(raw\_W)),
$$

再按 node depth 从 ancestor 到 descendant 调用：

```python
set_semantic_theta(node_id, target)
```

利用已有 path-additive semantic offset，使所有当前 node 初始时都精确满足：

$$
\tilde\theta_n^{sem}=\tilde\theta^{cold}.
$$

同步发生在三个位置：

1. backbone/tree 初次构造后；
2. `ConstructMemoryTree.cold_start_hawkes()` 完成后；
3. `Train.py` CLI 的 cold start 完成后。

### 35.3 修改结果

- cold start 现在真正决定 Memory semantic model 的初始点；
- 所有已有 internal/leaf node 与 cold-start Hawkes raw parameters 精确一致；
- 后续 node embedding、hypernetwork 和 semantic offset 仍保持可训练；
- 不需要把 standalone Hawkes `raw_mu/raw_W` 重复加入 Memory optimizer。

---

## 36. Sparse Retriever 的零梯度退化

### 36.1 问题

retriever 原本使用 entmax-1.5：

$$
\rho=\operatorname{entmax}_{1.5}(s/\tau).
$$

当 entmax 退化为单一 exact support 时：

1. support 内只有一个 memory；
2. normalized attention 的输出恒为 `1`；
3. 对 query、temperature、similarity scale、usage/age penalty 的局部梯度均可能为 `0`。

实际 NLL 梯度审计中，旧实现出现：

```text
query_net gradient = 0
retriever gradient = 0
连续两个 sequence 后 query_net parameter change = 0
```

这意味着 episodic correction 虽然进入 forward，但 retrieval policy 可能完全学不起来。

### 36.2 修改方法

保留 entmax sparse branch，同时增加一个小比例 dense gradient branch：

$$
\rho^{sparse}=\operatorname{entmax}_{1.5}(s/\tau),
$$

$$
\rho^{dense}=\operatorname{softmax}(s/\tau),
$$

$$
\rho
=(1-\epsilon_g)\rho^{sparse}
+\epsilon_g\rho^{dense},
\qquad \epsilon_g=0.05.
$$

最终 episodic attention 仍按原公式计算：

$$
\alpha_r
=\frac{\rho_r\exp(\gamma\,sim_r)}
{\sum_j\rho_j\exp(\gamma\,sim_j)+\epsilon}.
$$

诊断信息同时保留：

- `rho`：实际混合 gate；
- `sparse_rho`：纯 entmax gate；
- `dense_rho`：梯度保障 branch；
- `effective_k`：仍按 `sparse_rho` 的 support 统计。

### 36.3 修改结果

- sparse retrieval 的主要归纳偏置保留 95%；
- 即使 entmax 只选择一个 memory，query/retriever 仍有来自 prediction NLL 的梯度路径；
- 实际梯度审计中 query net 与 retriever 梯度从严格 `0` 变为非零；
- 连续 sequence 训练后 query net 参数产生实际变化；
- 原有 node-local retrieval 和 responsibility-weighted usage 测试继续通过。

---

## 37. Sleep Replay 绕过完整计算图

### 37.1 问题

旧 sleep objective 对每条 memory 直接计算：

$$
\ell^{replay}
\left(\tilde\theta_{node}^{sem}+\Delta\tilde\theta_r\right).
$$

这个 replay 能更新 semantic embedding、hypernetwork 和 offset，但没有经过：

```text
encoder → router → query net → retriever → effective parameter
```

因此 sleep loss 对 encoder、routing parameters、query net 和 retriever 没有实际训练作用。

### 37.2 修改方法

`compute_sleep_objective()` 现在对每个 stored `EventWindow`：

1. 恢复 full-history `times/types`；
2. 对 `start_idx:end_idx` 中每个 target event 单独 replay；
3. 使用与 wake 相同的 `_encode_memory_event()`，只编码 strict prefix；
4. 调用完整 `HawkesTree.forward()`：

   ```text
   encoder
   → node/global state
   → soft routing
   → all-path episodic retrieval
   → semantic/episodic fusion
   → effective Hawkes parameters
   ```

5. 用 effective parameters 计算 target event NLL；
6. `update_memory_state=False`，避免 replay 反复污染 usage；
7. sleep 不使用 sequence-local working adaptation，因此传入 zero working residual；
8. 先对单个 window 内事件取均值，再对 windows 取均值，保持原有 per-window normalization；
9. sleep gradient clipping 从仅 tree 扩展到 `tree + encoder`。

### 37.3 修改结果

sleep replay 的梯度现在能够实际到达：

- causal/attention encoder；
- tree routers；
- node embeddings 与 semantic offsets；
- shared hypernetwork；
- memory query net；
- sparse retriever parameters。

episodic `delta_theta` 仍然是 detached persistent buffer。这对应 practical non-parametric memory write；sleep 学习的是如何编码、路由和检索这些 residual，而不是让历史 memory tensor 持有跨 epoch autograd graph。

---

## 38. 保留的算法边界

本次没有把以下机制强制改成普通 backward，因为它们本身就是模型定义的一部分：

### 38.1 Working memory

working residual 仍按 inner gradient step 更新：

$$
\Delta\tilde\theta_t^{wm}
=\rho\Delta\tilde\theta_{t-1}^{wm}
-\eta_f\nabla_{\Delta\tilde\theta^{wm}}\ell_t.
$$

它是 sequence-local fast state，不注册为长期 `nn.Parameter`。

### 38.2 Episodic write

written residual 在 commit 时 detach 后存入 bank，属于 non-parametric update。未来 prediction loss 仍能通过 retrieval weights 更新 query/retriever，但不会反传穿过历史 insertion 操作。

### 38.3 Split / Merge / Prune

这些操作仍是 sleep-time structural edit。SplitModule 的 continuous proposal 会单独优化，commit 后再把新 router、node embedding 和 semantic offset 注册进主 optimizer。

---

## 39. 修改后的验证结果

### 39.1 完整测试

运行：

```bash
/Users/zimoshen/opt/miniconda3/envs/memTree/bin/python \
  -m unittest discover -s tests -v
```

结果：

```text
Ran 22 tests
OK
```

### 39.2 新增关键回归覆盖

新增测试验证：

1. prediction NLL 对 Attention encoder、query net、retriever 的梯度均非零；
2. sleep replay 对 encoder、router、query net、retriever 的梯度均非零；
3. cold-start Hawkes raw parameters 精确初始化所有 tree nodes；
4. training write 必须等待完整 future horizon；
5. 未完成 horizon 的 write 保持 pending；
6. delayed `QUEUE_SPLIT` 能真实 commit split；
7. split 新建的所有 tree parameters 均被主 optimizer 管理；
8. wake、sleep、checkpoint 和 wake-only inference 端到端通过。

### 39.3 CLI Smoke Test

使用两条三事件 sequence 完成：

```text
CSV load
→ Hawkes cold start
→ semantic initialization
→ wake training
→ sleep training
→ checkpoint save
```

全流程正常结束，无 non-finite loss 或 optimizer parameter omission。

### 39.4 最终结论

修复后，Memory 的连续可训练部分形成了严格因果的端到端链：

$$
\ell_t^{pred}
\rightarrow \tilde\theta_t^{eff}
\rightarrow
\left(
r_{t\ell},
\tilde\theta_\ell^{sem},
\alpha_{t,r},
q_t,
z_t
\right).
$$

与此同时，working-memory inner update、detached episodic insertion 和 sleep structural edit 继续保持各自明确的时间尺度与梯度边界。

---

# 次要功能断点修复记录（2026-07-13）

## 40. `lambda_write` 对训练梯度无作用

### 40.1 问题

旧 wake objective 在 hard action 之后添加常数：

$$
\lambda_{write}\mathbf 1[a_t=\mathrm{MEMORIZE}].
$$

该常数不依赖任何模型参数，因此不会改变 gradient；同时 `QUEUE-SPLIT` 也会产生 residual write，却没有被计入 write cost。

### 40.2 修改方法

两类 write action 共用条件：surprise 和 novelty 超过阈值。新增 smooth surrogate：

$$
p_t^{write}
=\sigma\left(\frac{s_t-\tau_s}{T_w}\right)
\sigma\left(\frac{n_t-\tau_n}{T_w}\right).
$$

再使用 straight-through gate：

$$
g_t^{ST}
=g_t^{hard}+p_t^{write}-\operatorname{stopgrad}(p_t^{write}),
$$

其中：

$$
g_t^{hard}
=\mathbf 1[a_t\in\{\mathrm{MEMORIZE},\mathrm{QUEUE\mbox{-}SPLIT}\}].
$$

所以 forward 中仍精确支付真实 hard write cost，backward 则通过 smooth surprise/novelty gate 更新 prediction 和 query 路径。

### 40.3 修改结果

- `MEMORIZE` 与 `QUEUE-SPLIT` 现在承担相同 write cost；
- `lambda_write` 对 surprise 和 novelty 均产生非零梯度；
- forward objective 仍与真实 hard action 数量一致；
- history 新增 `write_decisions`，与实际完成 commit 的 `writes` 分开统计。

---

## 41. Split Child 缺少 Hypernetwork Initialization Loss

### 41.1 问题

旧 split 直接通过 `semantic_offset` 把 child semantic parameter 精确设置为 `theta_cand`。这能保证结构提交正确，但 child embedding 与 shared hypernetwork 本身不必接近目标，semantic offset 会承担全部差异。

因此缺少截图定义的初始化拟合：

$$
\mathcal L_{init}
=\sum_{j=1}^{2}
\left\|
h_\psi(e_{\ell,j})
-\operatorname{stopgrad}(\tilde\theta_{\ell,j}^{target})
\right\|_2^2.
$$

### 41.2 修改方法

在 `HawkesTree` 新增 `fit_new_node_semantics()`。split 创建 child 并写入 parent target 后：

1. 固定 shared hypernetwork、ancestor embeddings 和已有 nodes；
2. 只优化两个新 child 的 local node embedding；
3. 最小化 child 当前 semantic output 与 detached `theta_cand` 的平方误差；
4. fitting 完成后，再调用 `set_semantic_theta()`；
5. semantic offset 只吸收剩余 approximation error，确保最终 child theta 仍与 proposal 精确相等。

配置项：

```text
split_init_steps
split_init_lr
```

Split 输出新增：

```text
init_loss_before
init_loss_after
```

### 41.3 修改结果

- child embedding/hypernetwork path 在 structural commit 前真实拟合 candidate theta；
- 测试确认 `init_loss_after < init_loss_before`；
- fitting 不会改变已有 node 的 semantic output；
- offset 校正后 child semantic theta 仍与 target 在数值精度内完全一致。

---

## 42. Checkpoint 无法 Resume Training

### 42.1 问题

旧 checkpoint 虽保存 `optimizer_state_dict`，但没有训练 loader，也没有保存 optimizer parameter-name mapping。动态 split 后 optimizer 会有新增 parameter groups；merge/prune 后还可能残留已移除参数，仅靠 PyTorch 的整数 parameter index 无法可靠映射到重建后的动态 topology。

此外还缺少：

- dormant SplitModule state/patience 的自动恢复；
- controller split queue；
- completed epoch；
- RNG state；
- resumed epoch 的确定性 data order。

### 42.2 修改方法

新增：

```python
MemoryTreeTrainer.from_checkpoint(...)
```

checkpoint format 升级为 version 3，并保存：

1. tree、Hawkes、encoder state；
2. dynamic topology 与 episodic memory extra state；
3. optimizer state；
4. 每个 optimizer group 对应的稳定 parameter names；
5. dormant split-module state 与 patience；
6. controller split queues；
7. wake/sleep/structure/training configs；
8. history 与 completed epoch；
9. CPU/CUDA RNG state。

每次 sleep transaction 和 checkpoint save 前都会 reconcile optimizer：删除 merge/prune 后已不属于模型的参数、保留有效 optimizer state、注册所有新增参数。

resume 后，epoch shuffle 使用 `seed + epoch - 1`，保证中断恢复时的数据顺序与连续训练一致。CLI 新增：

```bash
python -m Train.Train \
  --data-path data.csv \
  --resume checkpoints/memory_tree.pt \
  --epochs 10 \
  --checkpoint checkpoints/memory_tree_resumed.pt
```

其中 `--epochs` 表示本次 resume 继续执行的额外 epoch 数。

### 42.3 修改结果

- 动态 split 后的 checkpoint 能恢复全部 tree/encoder optimizer 参数；
- optimizer group 和 moment state 按稳定名称重新绑定；
- memory bank、split module、history 和 completed epoch 正确恢复；
- 恢复后训练从 epoch `N+1` 继续，不覆盖旧 history；
- API 与 CLI 均完成 epoch 1 保存、恢复、epoch 2 继续训练验证；
- 新增测试后完整结果为 `Ran 22 tests / OK`。
