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

---

# Router 训练塌缩保护修复记录（2026-07-24）

## 43. 问题背景与实验现象

### 43.1 前置修复

在本次修改之前，已经完成两项 Router 前置修复：

1. 导入不平衡 `H_tree` 拓扑后，根据左右子树的叶节点数量初始化
   Router bias，使 neutral routing 的初始叶节点先验从
   `2 ** (-depth)` 修正为 `1 / num_leaves`；
2. 将 `tau_similarity` 从 `0.7` 调整为 `0.5`，并强制检查：

   $$
   \tau_{\mathrm{similarity}}
   <
   1-\tau_{\mathrm{novelty}},
   $$

   避免 `QUEUE_SPLIT` 在数学上不可达。

这两项修改解决了 Router 的初始深度偏置和 split trigger 逻辑矛盾，
但不能阻止 Router 在梯度训练过程中再次偏向单一叶节点。

### 43.2 分层数据上的真实塌缩

使用每个 cluster 等量抽样的 stratified 数据训练后，得到：

```text
Epoch 1: H_marg=1.4608, max_mass=0.5009
Epoch 2: H_marg=0.3244, max_mass=0.9331
Epoch 3: H_marg=0.1016, max_mass=0.9868, actions=['leaf_prune']
Epoch 5: H_marg=0.0081, max_mass=0.9991
```

13 个叶节点的最大边际熵为：

$$
H_{\max}=\log 13\approx2.565.
$$

有效叶节点数量可用 entropy perplexity 表示：

$$
N_{\mathrm{eff}}=\exp(H_{\mathrm{marg}}).
$$

因此第 5 轮：

$$
N_{\mathrm{eff}}
=\exp(0.0081)
\approx1.008,
$$

说明几乎所有 soft responsibility 都进入同一个叶节点。这不是抽样偏差，
而是训练过程中发生的真实 Router collapse。

### 43.3 塌缩的原因

原 wake objective 为：

$$
L_{\mathrm{wake}}
=
L_{\mathrm{NLL}}
+L_{\mathrm{WM}}
+L_{\mathrm{write}}.
$$

其中没有约束不同叶节点的批量使用率。与此同时：

1. 所有叶节点初始时精确继承相同的 cold-start Hawkes 参数；
2. 早期不同叶节点具有近似相同的预测似然，Router 缺少可辨识信号；
3. 一个很小的数值优势会使某个叶节点获得更多 responsibility；
4. hard argmax 又使该叶节点获得更多 episodic memory；
5. 更多 memory 和梯度进一步强化该叶节点，形成正反馈；
6. low-mass prune 在 Router 稳定之前启动，会把 collapse 的结果误判为
   有意义的结构压缩。

原日志只报告边际熵和最大质量，无法区分：

- 所有样本都均匀且 Router 尚未专门化；
- 每个样本都很确定，但不同样本均衡地分配到不同叶节点；
- soft responsibility 尚均衡，但 hard argmax 已集中到单一叶节点。

---

## 44. 解决方法一：批量边际负载均衡

### 44.1 数学定义

对一个包含 $B$ 个独立序列前缀的 batch，Router 输出：

$$
r_i=(r_{i1},\ldots,r_{iL}),
$$

其中 $L$ 是当前叶节点数量。批量平均路由为：

$$
\bar r_\ell
=
\frac{1}{B}\sum_{i=1}^{B}r_{i\ell}.
$$

以当前叶节点上的均匀分布：

$$
u_\ell=\frac{1}{L}
$$

作为负载目标，定义：

$$
L_{\mathrm{balance}}
=
D_{\mathrm{KL}}(\bar r\|u)
=
\sum_{\ell=1}^{L}
\bar r_\ell
\log\frac{\bar r_\ell}{1/L}.
$$

校准阶段的完整目标为：

$$
L_{\mathrm{calibration}}
=
\frac{1}{B}\sum_{i=1}^{B}L_{\mathrm{NLL},i}
+\lambda_{\mathrm{balance}}L_{\mathrm{balance}}.
$$

默认：

```text
lambda_route_balance = 1.0
route_balance_batch_size = 64
```

### 44.2 为什么不能直接对单样本做 KL

本次约束的是：

$$
D_{\mathrm{KL}}
\left(
\frac{1}{B}\sum_i r_i
\middle\|
u
\right),
$$

而不是：

$$
\frac{1}{B}\sum_iD_{\mathrm{KL}}(r_i\|u).
$$

例如两个样本的责任度分别为：

$$
r_1=(0.9,0.1),\qquad
r_2=(0.1,0.9).
$$

它们的边际平均为：

$$
\bar r=(0.5,0.5),
$$

因此 batch balance penalty 为零。两个样本仍然可以分别对不同叶节点保持
高置信度，不会被强迫成逐样本均匀路由。

### 44.3 为什么使用独立的跨序列校准 pass

原 wake loop 是严格在线的：

```text
event forward
→ event backward
→ optimizer step
→ working-memory update
→ delayed episodic write
```

如果在该 loop 中跨 optimizer step 保留多个样本的计算图，会产生 stale graph，
并破坏 working/episodic memory 的在线时间语义。如果只在单条 sequence 内计算
边际，又会错误地要求单一 cluster 的一条序列局部使用全部叶节点。

因此每个 epoch 的 streaming wake 完成后，新增一次 Router calibration：

1. 打乱不同序列；
2. 每条序列随机抽取一个 target event；
3. Encoder 只读取该 event 之前的严格因果 prefix；
4. 每个 batch 至少包含两个独立序列；
5. 允许读取已有 episodic memory；
6. 禁止修改 memory usage、age、working state、episodic write 和 topology；
7. 在同一个计算图内联合计算 batch prediction NLL 和 marginal KL；
8. 完成一次正常 backward 和 optimizer step。

如果最后一个 batch 只有一个样本，会将其合并到前一个 batch，避免 marginal
KL 退化成单样本均匀约束。

### 44.4 方法合理性

该方法具有以下统计含义：

- `NLL` 保证 Router 仍然服务于预测目标；
- marginal KL 只惩罚整体负载塌缩；
- 单个样本仍可以形成低熵、高置信度路由；
- 对当前 tree split/merge 后的任意叶节点数量 $L$ 自动使用 `1/L`；
- 在本项目的 13-cluster 平衡数据上，均匀边际与经验 cluster prior 一致。

如果未来训练数据的真实 cluster prior 不均匀，目标 $u$ 应替换为经验先验
$\pi_\ell=N_\ell/N$，而不应继续机械使用均匀分布。

---

## 45. 解决方法二：Router 独立低学习率

### 45.1 原问题

原 optimizer 将 Router、semantic tree 和 causal encoder 放在同一个参数组，
都使用：

```text
learning_rate = 1e-3
```

Router bias 直接控制整条子树的概率质量。在不平衡二叉树中，一个高层 Router
bias 的小幅变化会同时改变多个叶节点的责任度，因此 Router 对相同学习率更敏感。

### 45.2 修改方法

optimizer 现在维护两个稳定参数组：

```text
base:
    lr = learning_rate

router:
    lr = learning_rate * router_lr_scale
```

默认：

```text
router_lr_scale = 0.1
```

即主学习率为 `1e-3` 时：

```text
router_lr = 1e-4
```

### 45.3 动态拓扑处理

Split 会动态创建新的：

- child node embedding；
- semantic offset；
- binary Router。

每次结构事务后，optimizer 会根据稳定参数名重新分组：

- `tree.routers.*` 始终进入低学习率 Router group；
- node embedding、semantic offset、hypernetwork 和 encoder 进入 base group；
- 被 merge/prune 删除的参数同时从 optimizer state 移除；
- 新增参数立即纳入 optimizer；
- checkpoint 按稳定参数名保存和恢复两个 parameter groups。

### 45.4 方法合理性

较低 Router 学习率不会改变模型的最优解，只改变优化动态：

- 降低一轮内 branch bias 快速跑偏的风险；
- 给 semantic parameters 和 encoder 足够时间形成可辨识差异；
- 减少早期随机优势被指数式放大的概率；
- 保留 Router 对长期预测梯度和 balance gradient 的响应能力。

---

## 46. 解决方法三：Leaf-Prune Warm-up

### 46.1 原问题

旧逻辑在每个 sleep cycle 都更新 low-mass streak。默认 patience 为 3，
所以 Router 一旦在前几轮短暂偏向某个叶节点，第 3 轮就可能执行：

```text
actions=['leaf_prune']
```

这会把尚未稳定的 Router 优化噪声永久写成 topology change。

### 46.2 修改方法

新增：

```text
prune_warmup_epochs = 10
```

前 10 轮：

```text
prune=off
```

这段时间内：

- 仍然计算和更新 leaf mass EMA；
- Split、Merge、Promotion 和 stale-memory pruning 仍可运行；
- 禁止 leaf-prune commit；
- low-mass streak 被重置为 0，不会在后台累计；
- 第 11 轮开启 prune 后，必须重新满足完整 patience。

### 46.3 方法合理性

该设计把“参数学习”和“不可逆结构删除”分离到不同时间尺度：

1. 先让 Router、semantic parameters 和 episodic memory 形成稳定分工；
2. 再根据持续低质量证据删除叶节点；
3. 避免把短暂训练波动误判为长期无用结构；
4. 不会阻止 split/merge 等其他有独立证据的结构操作。

---

## 47. 解决方法四：完整 Router 统计诊断

### 47.1 单样本平均条件熵

$$
H_{\mathrm{cond}}
=
\frac{1}{B}\sum_{i=1}^{B}
\left(
-\sum_{\ell=1}^{L}
r_{i\ell}\log r_{i\ell}
\right).
$$

它衡量单个样本的平均不确定性：

- 高：每个样本的路由接近均匀；
- 低：每个样本的路由更加确定。

日志字段：

```text
H_cond
```

### 47.2 边际熵

$$
H_{\mathrm{marg}}
=
-\sum_{\ell=1}^{L}
\bar r_\ell\log\bar r_\ell.
$$

它衡量整个 batch 对叶节点的总体利用率：

- 接近 $\log L$：soft load 总体均衡；
- 接近 0：soft responsibility 塌缩到单一叶节点。

日志字段：

```text
H_marg
```

### 47.3 Router Mutual Information

$$
I(\mathrm{sample};\mathrm{leaf})
=
H_{\mathrm{marg}}-H_{\mathrm{cond}}.
$$

日志字段：

```text
route_MI
```

解释：

- `H_cond ≈ H_marg ≈ log(L), MI ≈ 0`：
  负载均衡，但每个样本都接近均匀，Router 尚未专门化；
- `H_marg` 较高、`H_cond` 下降、`MI` 上升：
  不同样本开始稳定选择不同叶节点，是期望状态；
- `H_marg ≈ 0`：
  几乎所有样本进入同一个叶节点，发生 soft collapse。

### 47.4 Hard Assignment 数量

对每个样本：

$$
\hat\ell_i=\arg\max_\ell r_{i\ell}.
$$

统计每个叶节点成为 argmax 的次数，日志字段：

```text
hard=[count_1, ..., count_L]
```

其顺序与当前 epoch 的 `leaf_ids` 一致，checkpoint history 同时保存：

```text
hard_assignment_counts
marginal_leaf_mass
```

该指标能够发现一种 soft entropy 看不到的问题：所有 responsibility 都接近均匀，
但由于同一个叶节点始终多出极小数值，hard argmax 仍然全部落到该节点。

### 47.5 其他新增日志

```text
balance_KL
max_mass
prune=off/on
```

其中：

- `balance_KL`：跨序列 calibration batch 上的边际 KL；
- `max_mass`：边际平均中最大的叶节点概率；
- `prune`：当前 epoch 是否允许 leaf prune。

---

## 48. 有效性验证

### 48.1 自动化测试

修改后运行：

```bash
/Users/zimoshen/opt/miniconda3/envs/memTree/bin/python \
  -m unittest discover -s tests -v
```

结果：

```text
Ran 32 tests
OK
```

新增测试覆盖：

1. batch marginal KL 不会强迫每个样本均匀；
2. 互补的高置信度 assignment 可以得到零 balance penalty；
3. Router 初始和动态 split 后都使用 `0.1 × base_lr`；
4. leaf-prune warm-up 不执行 prune，也不累计 hidden streak；
5. checkpoint history 保存新增路由统计；
6. hard assignment 总数严格等于本轮事件数；
7. 原有 wake/sleep/split/merge/prune/checkpoint/inference 测试全部通过。

### 48.2 真实不平衡拓扑的分层 Smoke Test

使用真实 20-leaf 不平衡 `H_tree`，每个 cluster 等量抽取 2 条序列，
每条截取 8 个事件，训练 3 轮。理论最大边际熵为：

$$
\log20\approx2.996.
$$

得到：

```text
Epoch 1: H_marg=2.9923, max_mass=0.0556, prune=off
Epoch 2: H_marg=2.9696, max_mass=0.0673, prune=off
Epoch 3: H_marg=2.9546, max_mass=0.0718, prune=off
```

与修复前 13-leaf 实验中：

```text
H_marg: 1.4608 → 0.0081
max_mass: 0.5009 → 0.9991
```

相比，新的 marginal balance、低 Router 学习率和 prune warm-up 成功阻止了
短期 soft routing collapse。

### 48.3 有效性的边界

真实 smoke 同时观察到：

```text
H_cond ≈ H_marg
route_MI 较低
hard assignment 仍可能集中
```

这说明当前方法已经解决“全部 soft probability 进入一个叶节点”的问题，但不能
单独保证 Router 已经学出有意义的 cluster specialization。该结果并不否定本次
修改，反而说明新增的条件熵、互信息和 hard-count 日志能够区分：

1. soft load collapse；
2. 均匀但尚未专门化；
3. 均衡且具有样本区分能力。

后续合理目标不是继续无限增大 balance weight，而是在保持较高
`H_marg` 的同时，使 `H_cond` 逐渐下降、`route_MI` 上升，并检查 hard assignment
是否在不同 cluster 间形成稳定分工。

---

## 49. 配置、运行方式与 Checkpoint 使用

新增 CLI 参数：

```text
--route-balance-weight
--route-balance-batch-size
--router-lr-scale
--prune-warmup-epochs
```

13-cluster stratified smoke 推荐命令：

```bash
CUDA_VISIBLE_DEVICES=3 python -m Train.Train \
  --data-path ../Data/tree_13/hawkes_dataset_13_stratified_smoke.csv \
  --h-tree ../Data/tree_13/h_tree_13.pt \
  --hawkes-checkpoint Checkpoints/hawkes_backbone_tree13.pt \
  --node-dim 128 \
  --num-basis 2 \
  --epochs 10 \
  --max-events-per-sequence 64 \
  --route-balance-weight 1.0 \
  --route-balance-batch-size 65 \
  --router-lr-scale 0.1 \
  --prune-warmup-epochs 10 \
  --checkpoint Checkpoints/memory_tree13_guarded.pt \
  --device cuda
```

旧的：

```text
memory_tree13_stratified.pt
```

已经包含 collapse 后的 Router 和一次错误时机的 leaf prune，不适合继续正式训练，
但应保留为 collapse regression checkpoint。新的 guarded 训练必须从：

```text
hawkes_backbone_tree13.pt
+ h_tree_13.pt
```

重新开始。

### 49.1 本次涉及文件

- `Memory/Train/Train.py`
- `Memory/Sleep/Coordinator.py`
- `Memory/tests/test_train_inference.py`
- `Memory/tests/test_sleep_coordination.py`
- `Memory/Train/README.md`

### 49.2 本次修改的最终结论

本次修改没有通过逐样本均匀约束掩盖 Router 问题，而是：

1. 用跨序列 batch marginal KL 直接约束总体负载；
2. 保留 prediction NLL，允许样本级专门化；
3. 用较低 Router 学习率控制优化时间尺度；
4. 用 prune warm-up 防止早期错误结构删除；
5. 用条件熵、边际熵、互信息和 hard counts 区分不同类型的路由状态。

自动化测试和真实分层 smoke 证明该方法能够阻止此前观察到的短期 soft collapse；
新增统计同时明确揭示：防止 collapse 与学出有意义的 specialization 是两个不同问题，
后续训练必须分别评估。

---

## 50. 第四轮 Wake 训练出现非有限梯度

### 50.1 问题

13-cluster guarded smoke 的前三轮路由统计为：

```text
H_marg = 2.5213, 2.5531, 2.4449
max_mass = 0.1476, 0.1068, 0.1567
```

这些值接近 13 个叶节点的最大边际熵 `log(13) = 2.5649`，说明此前的 Router
soft collapse 已经得到控制。第四轮中断发生在 Wake 反向传播后的
`clip_grad_norm_`：

```text
RuntimeError: The total norm ... is non-finite
```

该异常有两种数学上不同的来源：

1. 某个梯度元素本身已经是 `NaN` 或 `Inf`；
2. 所有梯度元素仍然有限，但 float32 在计算
   `sqrt(sum(g_i^2))` 时先执行平方，`g_i^2` 溢出为 `Inf`。

旧代码只报告“总范数非有限”，无法区分这两种情况，也没有给出参数名、序列、
事件、叶节点和 action。直接改成 `error_if_nonfinite=False` 并不能解决问题：
当总范数为 `Inf` 时，缩放系数可能变成 0；当梯度含 `NaN` 时，还可能污染参数
和 Adam 状态。

此外，episodic residual write 原来直接持久化：

```text
Delta theta = P_r(-eta_mem * grad L_window)
```

但没有限制窗口梯度范数。一次高 surprise 窗口可能写入超大 residual；promotion
又会把已接受的子节点 memory 移到共享父节点，从而扩大该 residual 的作用范围。

### 50.2 解决方法

#### A. 有限性检查与数值稳定的全局梯度裁剪

新增 `clip_grad_norm_finite`，用于 Wake、Router calibration 和 Sleep：

1. 先逐元素检查所有已优化参数的梯度；
2. 若存在真实 `NaN/Inf`，立即停止并报告具体参数；
3. 若梯度均有限，先取
   `m = max_i |g_i|`；
4. 计算

   ```text
   ||g||_2 = m * sqrt(sum_i (g_i / m)^2)
   ```

   其中平方项都位于 `[0, 1]`，不会发生 float32 大数平方溢出；
5. 最后的标量乘法使用 float64，再按 `grad_clip` 缩放原梯度。

Wake 报错上下文现在包含：

```text
sequence, event, leaf, action, parameter name, NaN count, Inf count
```

每轮日志同时新增：

```text
grad_max=<该轮裁剪前的最大 Wake 全局梯度范数>
```

#### B. 约束持久化 episodic residual 的源梯度

`WakeObjectiveConfig` 新增：

```text
memory_write_grad_clip = 5.0
```

在低秩投影和持久化之前，先用 float64 计算写入窗口梯度范数并裁剪。默认配置下：

```text
||Delta theta_write||_2
<= eta_memory_write * memory_write_grad_clip
= 0.01 * 5
= 0.05
```

低秩正交投影不会增大 Frobenius 范数，因此该上界在投影后仍然成立。

#### C. 阻止非有限状态进入 memory

增加三层检查：

1. Working memory 拒绝非有限梯度，并用 float64 计算其裁剪范数；
2. episodic residual 在低秩投影后再次检查有限性；
3. `MemoryBank.add` 拒绝包含 `NaN/Inf` 的 key 或 residual。

这些检查保证错误在污染 checkpoint 之前暴露，而不是等到若干轮 retrieval 或
promotion 后才表现为模型参数异常。

### 50.3 方法合理性与有效性

这次修改没有降低 loss 权重、跳过异常 event，也没有关闭非有限检查，因此不会把
数值故障伪装成训练成功。

其合理性体现在：

1. 对“有限大梯度”仍执行方向不变的标准全局范数裁剪，只修正范数计算方式；
2. 对真实 `NaN/Inf` 继续 fail-fast，并提供可定位信息；
3. episodic residual 是跨事件、跨轮次的持久状态，其稳定性要求应严格高于普通
   单步参数梯度，因此写入前增加范数上界是合理的；
4. `0.05` 的默认 residual 范数上界与 `eta_memory_write=0.01`、
   `grad_clip=5.0` 一致，不引入新的任意量级；
5. promotion 只迁移 residual，不再能够把一次无界写入扩散到共享祖先。

新增回归测试覆盖：

1. 梯度元素为 `1e30`、普通 float32 范数会溢出的情况；
2. 梯度包含真实 `NaN` 时能报告参数名；
3. 超大 memory-write 源梯度在持久化后满足 residual 范数上界。

本地已完成 Python 语法编译检查。当前本地 Python 环境没有安装 PyTorch，因此
需要在 GPU 服务器的项目环境执行完整测试与 13-cluster resume smoke。

---

## 51. CUDA Checkpoint Resume 的 RNG State 类型错误

### 51.1 问题

服务器完成 14 个测试后，从第三轮 checkpoint 恢复时报错：

```text
TypeError: RNG state must be a torch.ByteTensor
```

模型、optimizer 和 history 均已成功加载；错误发生在最后的随机数生成器状态恢复。
原因是 checkpoint 使用：

```python
torch.load(checkpoint_path, map_location=device)
```

当 `device=cuda` 时，`map_location` 不仅移动模型张量，也把保存的 CUDA RNG state
移动到了 GPU。但 `torch.cuda.set_rng_state` 要求传入位于 CPU 的
`torch.uint8` ByteTensor，因此类型检查失败。

此外，保存 checkpoint 时可见的 GPU 数量可能与恢复时不同。例如保存时有 4 张卡，
恢复命令 `CUDA_VISIBLE_DEVICES=3` 只暴露 1 张卡；直接调用
`set_rng_state_all` 也不能安全处理这种设备数量变化。

### 51.2 解决方法

新增 `normalize_cuda_rng_states`：

1. 同时接受单个 Tensor、list 或 tuple；
2. 将每个 RNG state 显式转换成 CPU、`torch.uint8`、contiguous Tensor；
3. 根据 `torch.cuda.device_count()` 只保留当前可见设备能够恢复的状态；
4. 使用 `torch.cuda.set_rng_state(state, device=i)` 逐卡恢复。

保存新 checkpoint 时也显式对每个 CUDA RNG state 调用 `.cpu()`，避免状态位置
依赖保存时设备。

### 51.3 方法合理性与有效性

RNG state 只控制后续随机顺序和随机算子，不参与模型 forward，也不改变已保存的
参数、optimizer state、memory bank 或训练 history。将其转回 CPU ByteTensor
完全符合 PyTorch API 的输入约束。

只恢复当前可见 GPU 与已保存 GPU 的交集，可以同时支持：

```text
多卡保存 -> 单卡恢复
单卡保存 -> 单卡恢复
单卡保存 -> 多卡进程中的对应首卡恢复
```

对于当前命令，`CUDA_VISIBLE_DEVICES=3` 在进程内部只有逻辑设备 `cuda:0`，
因此恢复保存列表中的第一个可用 RNG state 是正确行为。

---

## 52. Retrieval Entmax 稀疏边界的 NaN Backward

### 52.1 问题

修复 RNG 恢复后，训练能够进入第四轮，并由新的梯度诊断精确定位到：

```text
wake sequence=16 event=55 leaf=root_R_R_L action=ASSIMILATE
```

非有限梯度同时出现在：

```text
episodic_memory.query_net
episodic_memory.retriever.raw_gamma/raw_tau/raw_lambda_*
encoder
```

而普通 Hawkes semantic 参数和 Router 并未同时报错。这种梯度拓扑说明 NaN 的共同
源头位于 `encoder -> query_net -> retriever` 的 retrieval 路径。

原来的 `entmax15_1d` 让 PyTorch 直接对阈值搜索过程求导：

```text
delta = clamp((1 - ss) / rho, min=0)
tau   = mean - sqrt(delta)
```

稀疏 entmax 正常工作时，大量未进入支持集的候选会满足 `delta=0`。虽然这些候选
对最终 `tau_star` 的上游梯度为 0，自动微分仍可能计算：

```text
0 * d sqrt(x)/dx | x=0
= 0 * Inf
= NaN
```

因此 forward 的 probability、loss 和 NLL 都可以完全有限，但 backward 会同时
污染 Retriever、QueryNet 和上游 Encoder。之前的 float32 norm 修复并没有制造
这个问题；它阻止了 optimizer 更新，并将原本不可定位的异常准确暴露出来。

### 52.2 解决方法

保留完全相同的 1.5-entmax forward 稀疏分布，但不再对 sort、support selection、
clamp 和 sqrt 阈值搜索自动求导。新增 `_Entmax15Function`，使用 1.5-entmax 的
闭式 Jacobian-vector product。

设最终概率为 `p`，并令：

```text
g_i = sqrt(p_i)
```

对于上游梯度 `v_i = dL/dp_i`，稳定 backward 为：

```text
dL/dz_i
= g_i * (v_i - sum_j(v_j g_j) / sum_j(g_j))
```

该表达式只依赖最终支持集概率：

1. 非活动位置 `p_i=0`，因此 `g_i=0`；
2. 不计算 `sqrt(delta)` 的导数；
3. 不对离散的排序和支持集大小求导；
4. 单元素支持时梯度自然为 0，不产生 NaN；
5. Retriever 原有的 5% dense-softmax 路径继续为极稀疏状态提供学习信号。

forward 同时先减去最大 logit。由于 entmax 对整体平移不变，这不改变输出，但会
降低中间量的数值范围。forward、backward 和上游梯度均新增明确的有限性检查。

### 52.3 方法合理性与有效性

该修改不是把 sparse entmax 替换为 softmax，也没有 detach 稀疏分支。它保留：

```text
完全相同的稀疏 forward
entmax 支持集上的精确梯度
原有 dense-gradient 辅助路径
```

变化仅在于使用数学闭式 backward，避免让通用 autograd 穿过非光滑的支持集搜索。
这正是稀疏概率映射在实现中应采用的方式。

新增边界回归用例使用：

```text
logits = [100, 0, -100]
```

该输入会产生单元素稀疏支持和多个 `delta=0` 的非活动候选。测试同时验证：

1. forward 概率有限且和为 1；
2. 输出确实只有一个非零支持；
3. backward 后所有 logit 梯度有限。

另外增加两项验证：

1. 在固定支持集的平滑区域用 double-precision finite-difference
   `gradcheck` 检查闭式 backward；
2. 构造极稀疏 `SmoothSparseRetriever`，验证 query 梯度以及
   `raw_gamma/raw_tau/raw_lambda_usage/raw_lambda_age` 梯度全部有限。

---

## 53. Parameter-independent Sequence Cache

### 53.1 问题

Memory Wake 按事件严格在线更新。对于长度为 `K` 的一条序列，旧实现第 `k` 个事件
重复执行：

1. 从 `times[:k]` 计算 `log(1+t)` 和 `log(1+delta_t)`；
2. 为 Hawkes intensity 重新计算所有历史事件的 exponential decay；
3. 为 `[t_{k-1}, t_k]` 重新计算所有历史事件的 kernel integral。

因此单序列上述部分的总历史扫描量为：

```text
0 + 1 + ... + (K - 1) = O(K^2)
```

13-cluster smoke 中 `65 x 64 = 4160` 个 Wake events，仅历史前缀位置就有：

```text
65 x 64 x 63 / 2 = 131040
```

而且相同数据在每个 epoch、Router calibration、memory write 和 Sleep replay 中会
再次计算。

### 53.2 解决方法

新增 `HawkesFamily.prepare_sequence_cache`，为每条序列保存三个只依赖数据和固定
decay basis 的张量。

#### A. Event time features

```text
f_k = [log(1 + t_k), log(1 + (t_k - t_{k-1}))]
```

`CausalPrefixEncoder` 仍然在每个事件使用当前可学习参数执行 embedding、
time projection 和 GRU，但直接切片缓存的 `f[:k]`，不再重复生成原始时间特征。

#### B. Hawkes decay sufficient statistics

对事件类型 `c` 和 basis `m`：

```text
S[k,c,m]
= sum_{j<k, type_j=c, t_j<t_k}
  exp(-decay_m * (t_k - t_j))
```

event intensity 改写为：

```text
lambda_d(t_k)
= mu_d + sum_{c,m} W[d,c,m] * S[k,c,m]
```

缓存保留旧实现的严格时间条件 `t_j < t_k`，因此相同 timestamp 的较早数组元素
不会错误地激发同一时刻事件。

#### C. Interval-integral sufficient statistics

对区间 `[t_{k-1},t_k]` 预计算：

```text
I[k,c,m]
= sum_{j<k, type_j=c}
  integral_{max(t_{k-1},t_j)}^{t_k}
  exp(-decay_m * (tau-t_j)) d tau
```

则：

```text
IntegralLoss_k
= sum_d mu_d * (t_k-t_{k-1})
 + sum_{d,c,m} W[d,c,m] * I[k,c,m]
```

缓存张量形状均为 `[K,D,M]`，13-cluster 的 `D=8, M=2, K=64` 时，每类 Hawkes
缓存只有 1024 个 float。

### 53.3 Cache 生命周期与覆盖范围

1. `MemoryTreeTrainer.train` 在 epoch 开始前一次性准备 dataset cache；
2. CPU cache 跨 epoch 保留，每条序列训练时只搬到 GPU；
3. Wake 和 Router calibration 使用同一 cache；
4. delayed memory write 直接复用当前序列 cache；
5. 新 `EventWindow` 保存截断到 `end_idx` 的三个 cache；
6. Sleep replay 和 shared replay likelihood 使用 window cache；
7. 旧 checkpoint 的 EventWindow 没有 cache 时，在第一次 replay 懒生成并回写；
8. `.to(device/dtype)` 会同步移动 window cache；
9. inference sequence 同样准备并使用 cache。

训练开始新增日志：

```text
[Cache] sequences=65 events=4160 built=65 time=...
```

### 53.4 方法合理性与有效性

这些 cache 不依赖任何会被 optimizer 更新的参数：

```text
依赖：times, types, fixed decays, num event types
不依赖：mu, W, Encoder, Router, QueryNet, Retriever, memory residual
```

所以即使每个事件的 effective `mu/W` 不同，仍可将当前 `W` 与固定 sufficient
statistics 相乘，NLL 的数学值及其对 `mu/W` 的梯度保持不变。

为避免 stale learned representation，本次没有缓存：

```text
GRU prefix embedding
Router responsibility
episodic retrieval output
effective Hawkes parameters
```

同时删除了 `compose_effective` 中 global QueryNet 的一次重复 forward，该修改只
消除未使用的重复计算，不改变输出。

新增测试覆盖：

1. 含相同 timestamp 的序列，缓存前后逐事件 NLL 总和一致；
2. 缓存前后对 raw mu/W 的梯度一致；
3. 缓存 intensity 与原始历史扫描一致；
4. 缓存 time features 前后 Encoder 输出一致；
5. 新写入 EventWindow 同时保存三类 cache。

本地 Python 语法编译检查已通过。完整 PyTorch 数值测试需要在 GPU 服务器环境运行。

---

## 54. Resume 后 Router 再次偏置且 Wake Write 为零

### 54.1 问题

混合版本 checkpoint 续跑到第5、6轮后出现：

```text
H_marg : 2.0360 -> 1.5484
max_mass: 0.3495 -> 0.5738
hard:
[1,0,0,0,0,0,4152,7,0,0,0,0,0]
-> [0,0,0,0,0,0,4160,0,0,0,0,0,0]
balance_KL: 0.7846 -> 1.3646
writes=memorize=queue_split=0
actions=[]
```

这里包含两个不同现象。

第一，Router 确实重新向第7个叶节点偏置。`hard=4160` 单独可能来自接近概率的
argmax，但 `H_marg` 同时下降、`max_mass` 和 `balance_KL` 同时上升，证明这不只是
hard tie，而是 soft marginal 也在塌缩。

第二，旧日志的 `actions=[]` 不是 Wake controller action。它只记录 Sleep transaction
中的 split/merge/promotion/prune。`writes=0` 只能证明没有 MEMORIZE/QUEUE_SPLIT，
无法区分事件是在 ASSIMILATE 还是 RETRIEVE。

### 54.2 根因

#### A. Wake 与 balance 更新次数严重不对称

当前 smoke 每轮有：

```text
4160 次逐事件 Wake optimizer step
65 个 calibration samples / batch_size 65
= 1 次 Router calibration optimizer step
```

即 balance correction 与 Wake 更新次数约为 `1:4160`。降低 Router 参数学习率只能
减慢 router linear layer 的漂移，但 Encoder 使用主学习率，并且 Router logits
直接依赖 Encoder 输出 `z_t`。因此 Encoder/Router 的联合表示仍可能在一轮 Wake
中偏移，而一次 epoch-end calibration 无法恢复。

#### B. 在旧优化规则 checkpoint 上续跑新规则

`memory_tree13_guarded.pt` 的前几轮状态来自：

```text
旧的自动微分 Entmax threshold backward
未限制范数的 episodic residual write
对应旧梯度分布的 Adam 一阶/二阶动量
旧规则生成并 promotion 的 memory bank
```

修复 Entmax 后，forward 语义相同，但 backward 已改为正确的闭式 Jacobian；
memory write 也新增了范数上界。继续恢复旧 Adam moments 和旧 memory state 虽然
在工程上可以加载，却不再是同一训练算法下的连续优化轨迹，适合作为问题复现，
不适合作为正式结果。

#### C. Write 为零尚不能直接归因

Controller 条件为：

```text
surprise <= tau_s                  -> ASSIMILATE
surprise > tau_s and novelty<=tau_n -> RETRIEVE
surprise > tau_s and novelty>tau_n
  and count<tau_c                  -> MEMORIZE
otherwise                          -> QUEUE_SPLIT
```

当前缺少 ASSIMILATE/RETRIEVE 和 novelty/max-sim 统计。很可能旧 memory keys 与当前
query 的最大相似度已经使 `novelty <= 0.3`，从而大量 RETRIEVE；但在加入诊断前
不能把该推断当成结论。

### 54.3 解决方法

#### A. 自适应多步跨序列 balance calibration

新增：

```text
route_balance_max_steps = 8
route_balance_target_kl = 0.1
```

每轮 Wake 后重复完整跨序列 calibration：

```text
最多8次
当最终 batch-marginal KL <= 0.1 时提前停止
```

它仍然约束：

```text
KL(mean batch responsibility || uniform leaves)
```

而不是逐样本均匀，因此不会直接禁止样本级 specialization。平衡状态下通常只执行
一次；出现路由偏置时才增加纠正强度。日志改为：

```text
balance_KL=<final value>@<used steps>
```

#### B. 完整区分 Wake 与 Sleep actions

Wake 日志新增：

```text
wake_actions=A.../R.../M.../Q...
novelty=<mean novelty>
max_sim=<mean maximum key similarity>
```

其中：

```text
A = ASSIMILATE
R = RETRIEVE
M = MEMORIZE
Q = QUEUE_SPLIT
```

旧的 `actions=[]` 重命名为：

```text
sleep_actions=[]
```

避免将“没有发生结构事务”误读为“没有 Wake memory action”。

checkpoint history 同时保存四类 action count、平均 novelty、平均 max similarity
和平均 similarity count。

### 54.4 方法合理性与有效性

自适应 calibration 没有对单个样本施加 uniform penalty，也没有用 hard assignment
作为可微目标；它只是让原有 batch-marginal KL 的优化频率与数千次 Wake update
不再相差四个数量级。

`target_kl=0.1` 对13个叶节点意味着 calibration marginal entropy 目标约为：

```text
H_marg >= log(13) - 0.1 = 2.4649
```

这仍允许每个样本具有尖锐且不同的路由，只要求总体负载不过度集中。

当前第5/6轮 checkpoint 已经具有明显 soft bias，并混合两套 backward/write
规则，不建议作为正式训练起点。应保留为 collapse regression checkpoint；最终
训练从 cold-start Hawkes 与 `h_tree_13.pt` 在统一的新代码下重新开始。

## 55. 三时间尺度训练边界与 sequence-level Router MI（2026-07-24）

### 55.1 问题

最新 clean run 虽然没有 NaN/Inf，且 sequence cache 正常生效，但第一轮即出现：

```text
wake_actions=A2415/R1739/M6/Q0
novelty=0.0043 max_sim=0.9957
balance_KL=1.1902@8
hard=[4154, 4, 0, ..., 1, 0]
```

它说明数值稳定性修复并没有解决训练时间尺度混合：

1. `train_wake_sequence()` 对每个 event 调用一次全局
   `optimizer.step()`。一条64-event sequence 会连续修改 Encoder、Router、
   semantic/hypernetwork、query/retriever 64 次；
2. 同一 sequence 内的 event 梯度高度相关，不是能够相互抵消的独立随机梯度。
   下一条 sequence 看到的是已被上一条 sequence 连续修改过的模型；
3. working memory 和长期参数复用了同一 event loss graph。原本只应承担快速、
   sequence-specific 适应的 `Delta theta_wm` 没有形成真正的隔离层；
4. Router 同时收到逐 event prediction NLL 和少量 epoch-end balance 更新。
   某叶偶然领先后会得到更多 NLL 梯度，形成
   `r_up -> more gradient -> faster NLL improvement -> r_up` 的正反馈；
5. 原 balance 只约束 event responsibility 的 batch marginal。它可以提高总体
   熵，却不能同时保证“不同 sequence 分到不同叶”和“单条 sequence 有明确归属”；
6. hard memory target 使用确定性 `argmax`。概率接近时列表中的第一个叶会永久
   赢得平局，使早期 memory keys 进一步集中。

因此论文设计的三种尺度

```text
working memory: event-level fast
episodic/shared adaptation: sequence-batch medium
semantic tree/topology: sleep-level slow
```

在旧实现中退化成了所有共享状态的 event-level 快速更新。

### 55.2 解决方法

#### A. Wake 只更新 working memory

Wake 进入 sequence 时临时冻结全部持久参数：

```text
stopgrad(
  Encoder,
  Router,
  semantic tree / hypernetwork,
  episodic query and retriever
)
```

每个 event 只计算：

```text
grad_wm = d(L_pred + lambda_wm ||Delta theta_wm||^2) / d(Delta theta_wm)
Delta theta_wm <- rho * Delta theta_wm - eta_wm * grad_wm
```

Wake 不再调用全局 `loss.backward()` 或 `optimizer.step()`。代码在进入 Wake
前清空旧梯度，单元测试验证 Wake 结束后所有持久参数逐元素不变且
`parameter.grad is None`。

memory action 和延迟写入仍然在线执行，因为它们属于观测驱动的 memory state
变化；但它们不再借用 working-memory 的 event graph 更新共享网络。

#### B. 跨 sequence 的全局 prediction update

新增 `train_global_batch_epoch()`。它在多个 sequence 完成 Wake 后重新执行
strict-causal forward：

```text
L_pred = (1 / total_events) * sum_s sum_t NLL(s,t)
```

实现按 sequence 构建并反向、只累积梯度，不执行 step；一个 sequence 的 graph
释放后再处理下一个。由于这期间参数保持不变，它与把整个 cross-sequence batch
一次性放入显存所得的 batch gradient 数学等价，但显存复杂度从“整个 batch 的
graph”降为“单条 sequence 的 graph”。

直到 batch 完成才进行一次：

```text
clip_grad_norm
global_optimizer.step()
```

当 `route_balance_batch_size=65` 且 smoke dataset 有65条 sequence 时，每轮只有
一次普通持久参数更新，而不是4160次。

#### C. Router 改为 sequence-level mutual information

对每条 sequence 聚合完整 event responsibility：

```text
r_bar_s = (1 / T_s) * sum_t r_{s,t}
r_bar   = (1 / |B|) * sum_s r_bar_s
```

定义：

```text
I(S;L) = H(r_bar) - (1 / |B|) * sum_s H(r_bar_s)
```

Router 全局目标改为：

```text
L_global =
    L_pred
    - lambda_route_mi * I(S;L)
    + lambda_route_prior * KL(r_bar || Uniform)
```

其中 MI graph 对 Encoder state 做 detach，只更新 Router；Encoder 从全局
prediction NLL 学习。这样：

- 增大 `H(r_bar)`，阻止整个 batch 全进同一叶；
- 减小平均 `H(r_bar_s)`，允许每条 sequence 有明确归属；
- prior KL 只控制总体负载，不强迫每条 sequence 都均匀。

对当前13个等规模 cluster，uniform prior 是合理的初始先验。未来若 cluster
本身不均衡，可将其替换为 Dirichlet-smoothed empirical prior，而不改变 MI 项。

#### D. causal sequence route 用于 memory assignment

Wake 在第 `t` 个 event 使用不包含未来信息的累计责任：

```text
r_bar_{s,t} = (1 / t) * sum_{tau<=t} r_{s,tau}
```

并按该分布采样 memory leaf，而不是确定性 `argmax`。这既使一条 sequence 的
memory 选择具有一致的历史依据，也消除了近似平局时“第一个叶永远胜出”的索引
偏置。checkpoint 保存的 CPU/CUDA RNG state 继续保证恢复运行的随机轨迹可复现。

#### E. Sleep 切断 Router prediction 梯度

Sleep replay 仍更新 Encoder、semantic/hypernetwork、episodic query/retriever，
但 effective Hawkes mixture 使用 detached routing weights。Router 只接受
sequence-level 全局目标，避免 Sleep NLL 重新引入 winner-take-more feedback。

#### F. 日志与 CLI

新日志字段：

```text
wm_grad_max       event-level working-memory 最大梯度范数
global/event      cross-sequence prediction NLL
global_steps      本轮持久参数 optimizer step 数
prior_KL          sequence marginal 对先验的 KL
seq_H_cond        平均单 sequence 路由熵
seq_H_marg        sequence batch 边际熵
seq_MI            H_marg - H_cond
seq_hard          每条 sequence 的 hard assignment 计数
mem_assign        每个 event 实际采样的 memory leaf 计数
```

新增 `--route-mi-weight`。Router 不再逐 event 更新，所以
`--router-lr-scale` 默认从 `0.1` 改为 `1.0`。
`--route-balance-max-steps` 和 `--route-balance-target-kl` 为兼容旧 shell
仍可解析，但不再触发重复 calibration。

### 55.3 方法合理性

1. **梯度隔离严格成立**：Wake graph 中只有临时 `working_delta` 可导；
   持久参数被冻结，消除了同一 event loss 的双重更新路径。
2. **跨序列梯度是无偏 batch 梯度**：只要累积期间不 step，
   `sum_s backward(L_s)` 与 `backward(sum_s L_s)` 对参数的一阶梯度完全相同。
3. **MI 比单独最大熵更适合无监督 Router**：仅最大化边际熵可能让所有样本都取
   相同均匀分布；MI 同时追求 batch 多样性和单 sequence 确定性。
4. **因果性保持**：在线 memory assignment 使用 `tau<=t` 的累计责任，不读取
   sequence 未来；完整 `r_bar_s` 只用于 epoch 内离线全局训练。
5. **时间尺度恢复**：64-event sequence 现在是
   `64次 forward + 64次 WM update + 0次 global step`；65条 sequence 汇总后才
   执行一次 global step，Sleep 再进行一次慢速 replay/structure cycle。

### 55.4 验证

本地测试结果：

```text
python -m unittest tests.test_train_inference
Ran 23 tests ... OK

python -m unittest discover -s tests -p 'test_*.py'
Ran 44 tests ... OK
```

新增测试覆盖：

- sequence MI 对“互补且明确”的路由高于“所有 sequence 都均匀”的路由；
- Wake 后持久参数完全不变，只有 working memory 发生更新；
- 两条 sequence、batch size 2 时只执行一次 global optimizer step；
- Sleep replay 对 Router 无梯度，但 Encoder/query/retriever 仍有梯度；
- history 的 `seq_hard` 总数等于 sequence 数，`mem_assign` 总数等于 event 数。

旧的 `memory_tree13_clean_v3_epoch1.pt` 已包含 collapse 状态，不能用于验证新训练
算法。下一轮仍应从 Hawkes cold-start checkpoint 与 `h_tree_13.pt` 重新开始。

## 56. Router 对称驻点、leaf expert 同质性与 likelihood-aligned routing（2026-07-24）

### 56.1 问题

三时间尺度修复后的前两轮日志为：

```text
epoch1:
prior_KL=-0.0000
seq_H_cond=2.5649 seq_H_marg=2.5649 seq_MI=0.0000
max_mass=0.0769
seq_hard=[65,0,...]
mem_assign≈各叶均匀

epoch2:
prior_KL=0.0000
seq_H_cond=2.5649 seq_H_marg=2.5649 seq_MI=0.0000
max_mass=0.0777
seq_hard=[0,0,0,0,0,0,65,0,...]
mem_assign≈各叶均匀
```

这不是 soft load collapse。`max_mass≈1/13` 且 `mem_assign` 均匀，说明平衡先验和
causal sampling 有效；问题是所有 sequence 的 soft responsibility 几乎完全相同。
`seq_hard` 整体从 leaf0 跳到 leaf6 只是近似平局时某个叶比其他叶大一点，不是
获得了 sequence cluster。

代码中存在两个严格对称源：

1. `_ensure_router()` 将所有 Router weight 初始化为零。即使 Encoder 产生不同
   `z_s`，初始 logit 也只含 branch bias，Router 对输入无依赖；
2. `initialize_semantics_from_hawkes()` 使用 `semantic_offset` 将每个 node 的有效
   Hawkes 参数精确写成同一个 cold-start target。13个 leaf expert 初始 likelihood
   完全同质。

在所有 sequence 路由相同的点：

```text
I(S;L)=0
grad I(S;L)=0
KL(mean route || uniform)=0
grad KL=0
```

单纯增大 MI weight 无法离开一阶梯度为零的驻点。此外，MI 只要求不同 sequence
获得不同路由，本身没有说明应该依据哪一种 Hawkes dynamics 划分；随机扰动后若
缺乏 likelihood 对齐，MI 可能放大任意随机分组。

### 56.2 解决方法

#### A. balanced prior + small input-dependent Router perturbation

`_ensure_router()` 不再把 weight 置零，而使用：

```text
xavier_normal_(router.weight, gain=0.05)
router.bias = 0
```

导入不平衡 H-tree 后仍调用 `initialize_router_bias_from_leaf_counts()`，所以 bias
继续表达左右子树叶数先验。新增 `initialize_router_weights(gain, seed)` 只重新
初始化 weight、绝不修改 bias，从而同时满足：

```text
总体边际先验接近均衡
不同 z_s 获得微小但非完全相同的 route
初始化可由 seed 复现
```

#### B. leaf-local、逐坐标零均值 Hawkes 扰动

在 exact cold-start 和 semantic smoke 之后、新建 optimizer 之前调用：

```text
break_initial_leaf_symmetry(relative_scale=0.01, seed)
```

对每个 raw Hawkes 参数坐标生成扰动并跨 leaf 去均值：

```text
theta_leaf(0) = theta_cold + epsilon * xi_leaf
sum_leaf xi_leaf = 0
```

只通过 leaf 自己的 `semantic_offset` 写入扰动。root 和 internal node 继续保持
精确 cold-start；所有 leaf raw 参数的算术平均仍等于 cold-start。因此在 uniform
route 下，raw-space 整体预测保持不变，但单个 leaf 已具有微小 likelihood 差异。

启动时输出：

```text
[Symmetry]
router_gain
router_norm_mean
leaf_scale
leaf_delta_mean
leaf_mean_error
spectral_radius=[min,max]
```

若任一 leaf 扰动后的 Hawkes branching spectral radius 大于等于1，启动立即失败，
要求降低 `--leaf-symmetry-scale`。

#### C. likelihood-aligned leaf routing

对 sequence `s`、leaf `l`，使用 cached Hawkes sufficient statistics 向量化计算：

```text
E_sl = mean_t NLL(event s,t | semantic_l + episodic_l)
```

新增：

```text
L_mix = -mean_s log sum_l [
    r_bar_sl * exp(-E_sl / route_energy_temperature)
]
```

全局目标变为：

```text
L_global =
    L_prediction
    + lambda_route_mix * L_mix
    - lambda_route_mi * I(S;L)
    + lambda_route_prior * KL(mean_s r_bar_s || Uniform)
```

`L_mix` 同时向 Router 和 leaf expert 提供梯度，使 sequence 更偏向能够解释其
Hawkes dynamics 的 leaf，避免 MI 只放大任意随机簇。

实现没有对13个 leaf 逐个重新扫描 history。它直接使用现有
`HAWKES_HISTORY_STATS` 和 `HAWKES_INTERVAL_STATS`，一次得到 `[L]` leaf event
energies。

#### D. 分阶段 MI-to-Encoder 梯度

前 `route_encoder_warmup_epochs=2` 轮：

```text
MI -> Router
MI -X-> Encoder
```

让小 Router/leaf 非对称性先形成稳定方向。后续使用 straight-through gradient
scaling：

```text
z_route = stopgrad(z) + alpha * (z - stopgrad(z))
alpha = 0.1
```

forward route 完全不变；Router 保留完整 MI 梯度，Encoder 只接收0.1倍 MI 梯度。
Encoder 在所有阶段都仍从 global prediction NLL 和 likelihood mixture 学习。

#### E. 新诊断

日志增加：

```text
mix             likelihood-aligned mixture objective
seq_route_std   sequence responsibility 跨样本标准差
mi_enc_scale    当前 MI 进入 Encoder 的梯度比例
```

若 `seq_route_std` 长期接近0，则 Router 仍未形成 input dependence；若
`seq_route_std` 上升但 `mix`/prediction 变坏，说明随机分工没有对齐 likelihood；
若 `seq_MI` 上升同时 prior KL 保持小且 mix 下降，才是合理 specialization。

### 56.3 为什么未改成 root-only

“root cold start -> memory accumulation -> sleep split”更符合动态生长树的纯论文
路线，但它与当前实验的明确输入 `h_tree_13.pt` 和预构建13叶 topology 不兼容。
若设置 root-only，就不能同时声称完整使用了 H-tree 的13叶结构。

本轮保留13叶作为 topology prior，并修复其对称性。root-only 应作为独立 ablation：

```text
A: H-tree initialized 13-leaf prior
B: root-only dynamic split
```

不应在同一 checkpoint/实验中混合两种定义。

### 56.4 合理性与验证

1. Router weight 是零均值小扰动，bias 仍保留 leaf-count balanced prior；
2. leaf raw parameter perturbation 逐坐标零均值，因此 uniform raw mixture 保持
   cold-start，不引入整体方向漂移；
3. likelihood mixture 给无监督分工提供 Hawkes dynamics 依据，没有使用 CSV
   `cluster` 标签，训练仍为 self-supervised/unsupervised；
4. MI-to-Encoder warmup 防止早期随机 Router 迅速扭曲表示，后期弱梯度允许
   Encoder 学到更适合分工的表征；
5. 新测试验证 Router weight 非零但 bias 不变、leaf 均值保持且 internal node
   不变、向量化 leaf energy 与逐 leaf cached Hawkes NLL 一致，以及
   MI-to-Encoder 梯度的0/0.1阶段隔离。

最终验证：

```text
python -m unittest discover -s tests -p 'test_*.py'
Ran 48 tests ... OK
```

2-sequence/4-event 端到端 CLI smoke 同时确认：

```text
[Symmetry] leaf_mean_error=0.000e+00 spectral_radius=[0.213413,0.213452]
global_steps=1
seq_route_std=6.640e-04
mi_enc_scale=0.00
seq_hard=[1,1]
```
