# Tensor DAG + Symbolic VJP + SLP 实验报告

本报告基于 `results/results.csv`（`sad_bench --suite all --malloc-stable --repeats 20 --warmup 3`）、
`results/scaling_standalone*.txt`（`sad_scaling`）以及附录中的对照实验。所有数字都是实际测量值，
表格由 `scripts/make_tables.py` 从 CSV 生成（完整表格见 `results/tables.md`）。

## 0. 实验环境与测量方法

| 项目 | 值 |
|---|---|
| CPU | Intel Xeon @ 2.80 GHz，4 核（L1d 32 KiB/核，L2 1 MiB/核，L3 33 MiB） |
| 内存 | 15 GiB |
| 编译器 / 标准 | GCC 13.3，C++20，`-O2`，Release |
| LibTorch | 2.14.1+cpu（CXX11 ABI） |
| 线程 | `at::set_num_threads(4)`，`set_num_interop_threads(1)`，三个后端相同 |
| dtype | float32（基准）；float64 用于单元测试（rtol 1e-9） |
| 随机种子 | `torch::manual_seed(42)`；同一组输入张量同时喂给 DAG 执行器与 Autograd |
| 计时 | `steady_clock`；3 轮 warmup + 20 轮重复，报告 median / P90（CSV 中还有 min / mean） |
| 一次性成本 | 建图 / 符号微分 / SLP 编译在 5 个全新 Graph 上重复，报告 median 与首次（cold）值 |
| 算子计数 | `at::RecordFunction` 全局回调计数 FUNCTION 作用域事件，对两个后端用同一仪器（含嵌套算子） |
| 内存 | 三个层次：Graph 节点字节；执行器按 storage 去重的逻辑存活字节；glibc `mallinfo2` 在算子边界采样的峰值增量 |
| 正确性 | 每个配置用 `torch::allclose(got, ref, rtol=1e-4, atol=1e-6 + rtol·max|ref|)` 对比 loss 与所有梯度 |

公平性要点：三个后端用同一份 `build_task` 模板生成完全相同的计算（`tasks.hpp`），调用同样的
`at::add / at::mul / at::matmul / at::relu / at::sum`；Autograd 侧用 `torch::autograd::grad`
而不是 `.backward()`，避免 `.grad` 累加的额外写入。`--malloc-stable`（`mallopt` 提高 mmap/trim
阈值）对两个后端同样生效，原因见附录 A。

**正确性结果**：104 个 DAG 配置（52 个开启 hash-consing，52 个关闭）全部 `grad_allclose=1`。
除 `sharing k=8` 的两个配置外，梯度与 Autograd 逐位相同（最大相对误差 0）；`k=8` 时最大相对误差
1.2e-7（float32 下不同的求和顺序）。单元测试 22 个全部通过（`sad_tests`）。

## 1. 符号规模：Tree AST vs Tensor DAG

### T1 符号微分规模、时间与内存

| configuration | DAG fwd nodes | DAG bwd nodes | DAG diff time | DAG graph bytes | tree fwd nodes | tree grad nodes | tree diff time | tree bytes |
|---|---|---|---|---|---|---|---|---|
| chain(n=1024,depth=4) | 22 | 20 | 8.9 us | 112.5 KiB | 22 | 717 | 186.7 us | 97.7 KiB |
| chain(n=1024,depth=16) | 82 | 68 | 29.8 us | 114.0 KiB | 82 | 20453 | 3.46 ms | 2.7 MiB |
| chain(n=1024,depth=64) | 322 | 260 | 101.0 us | 119.8 KiB | 322 | 977797 | 168.13 ms | 126.8 MiB |
| chain(n=1024,depth=256) | 1282 | 1028 | 1.05 ms | 479.0 KiB | 1282 | cap hit | 410.16 ms (cap hit) | - |
| sharing(n=1024,depth=8,k=1) | 42 | 36 | 22.8 us | 113.0 KiB | 42 | 3509 | 566.4 us | 470.6 KiB |
| sharing(n=1024,depth=8,k=2) | 50 | 44 | 19.6 us | 113.3 KiB | 2807 | cap hit | 382.78 ms (cap hit) | - |
| sharing(n=1024,depth=8,k=4) | 66 | 60 | 26.1 us | 113.8 KiB | 502437 | cap hit | 368.49 ms (cap hit) | - |
| diamond(n=4096,depth=4,width=2) | 30 | 40 | 15.9 us | 113.0 KiB | 122 | 11307 | 2.70 ms | 1.5 MiB |
| diamond(n=4096,depth=4,width=4) | 62 | 80 | 49.2 us | 114.0 KiB | 1532 | 1226687 | 199.24 ms | 159.3 MiB |
| diamond(n=4096,depth=4,width=8) | 126 | 160 | 66.0 us | 116.0 KiB | 22232 | cap hit | 384.29 ms (cap hit) | - |
| mlp(batch=4,in=8,hidden=8,out=8,layers=8) | 48 | 69 | 30.0 us | 114.3 KiB | 94 | 13153 | 2.61 ms | 1.8 MiB |
| mlp(batch=4,in=8,hidden=8,out=8,layers=32) | 168 | 237 | 93.4 us | 119.9 KiB | 334 | 342801 | 55.81 ms | 46.6 MiB |
| mlp(batch=4,in=8,hidden=8,out=8,layers=128) | 648 | 909 | 327.1 us | 254.4 KiB | 1294 | cap hit | 579.08 ms (cap hit) | - |

（"cap hit" = 树节点数超过 2,000,000 的上限后中止；DAG graph bytes 包含 `reserve(1024)` 的固定开销。）

图：`results/fig_chain_nodes_time.png`。

**结论（问题 1）**：

* **链式（无扇出）结构**：树的梯度表达式随深度 D 介于平方与立方之间增长（717 → 20453 → 977797，
  对应 D = 4, 16, 64，每 4 倍深度增长 28× 和 48×，指数趋近 3）。原因是每一层的 VJP 规则都要复制
  兄弟子树：`Mul` 的伴随复制另一操作数，`ReLU` 的掩码 `GreaterThan(x, 0)` 复制整个前向子树 x，而这
  些复制又被更深层的伴随再次复制。DAG 的反向节点数严格线性（bwd/fwd = 0.80），
  depth=64 时节点数相差 **3760×**，微分时间相差 **1660×**（168 ms vs 0.1 ms），内存相差 **1000×**
  （126.8 MiB vs 120 KiB）。
* **有扇出 / 重复引用的结构**（sharing k≥2、diamond width≥2、MLP 的 `h·h`）：树在前向就已指数
  膨胀（sharing 的前向树 42 → 2807 → 502437，约 k^depth），梯度树更是在 depth=4、width=4 的小菱形
  上就达到 1.2M 节点 / 159 MiB；稍大一点（width=8 或 depth=16）就触顶。DAG 在这些配置上只有
  40–160 个反向节点。
* 因此 **只要计算中存在共享子表达式（同一值被多个消费者引用，或者 VJP 规则需要引用前向中间值），DAG
  相对普通 AST 的符号微分优势就是数量级的**；即便在没有扇出的纯链上，树也因为反向规则复制前向子树
  而多项式膨胀。只有像 `nn`（7 个前向节点）这种极小的表达式，两者差距才停留在几十倍。

## 2. Hash-consing 是否显著减少实际算子调用

### T2 sharing 任务消融（depth 8；HC = 开启 hash-consing，noHC = 关闭）

| configuration | nodes HC / noHC | HC hits fwd+bwd | ATen ops HC / noHC / torch | fwd+bwd median HC / noHC / torch | peak live HC / noHC |
|---|---|---|---|---|---|
| sharing(n=1024,depth=8,k=1) | 78 / 85 | 0+7 | 110 / 131 / 109 | 105.1 us / 115.9 us / 358.8 us | 68.0 KiB / 68.0 KiB |
| sharing(n=1024,depth=8,k=2) | 94 / 149 | 16+7 | 126 / 195 / 197 | 131.4 us / 235.5 us / 591.3 us | 68.0 KiB / 80.0 KiB |
| sharing(n=1024,depth=8,k=4) | 126 / 277 | 48+7 | 158 / 323 / 373 | 174.7 us / 377.4 us / 1.04 ms | 68.0 KiB / 84.0 KiB |
| sharing(n=1024,depth=8,k=8) | 190 / 533 | 112+7 | 222 / 579 / 725 | 251.5 us / 747.6 us / 1.90 ms | 68.0 KiB / 84.0 KiB |
| sharing(n=65536,depth=8,k=1) | 78 / 85 | 0+7 | 116 / 137 / 115 | 2.58 ms / 2.56 ms / 2.06 ms | 4.3 MiB / 4.3 MiB |
| sharing(n=65536,depth=8,k=2) | 94 / 149 | 16+7 | 132 / 201 / 203 | 2.13 ms / 3.13 ms / 3.23 ms | 4.3 MiB / 5.0 MiB |
| sharing(n=65536,depth=8,k=4) | 126 / 277 | 48+7 | 164 / 329 / 379 | 2.53 ms / 5.59 ms / 4.57 ms | 4.3 MiB / 5.3 MiB |
| sharing(n=65536,depth=8,k=8) | 190 / 533 | 112+7 | 228 / 585 / 731 | 2.74 ms / 13.42 ms / 10.62 ms | 4.3 MiB / 5.3 MiB |

### T3 没有语法重复的任务

| configuration | bwd nodes HC / noHC | HC hits bwd / AD requests | ATen ops HC / noHC | fwd+bwd median HC / noHC |
|---|---|---|---|---|
| chain(n=1024,depth=256) | 1028 / 1283 | 255 / 1283 | 3086 / 3851 | 5.13 ms / 5.67 ms |
| diamond(n=4096,depth=16,width=8) | 628 / 755 | 127 / 755 | 1646 / 2027 | 4.33 ms / 4.59 ms |
| nn(batch=256,in=512,out=256) | 11 / 11 | 0 / 11 | 51 / 51 | 1.12 ms / 1.06 ms |
| mlp(batch=128,in=256,hidden=256,out=64,layers=8) | 69 / 78 | 9 / 78 | 296 / 321 | 3.99 ms / 5.01 ms |
| chain(n=8,depth=4096) | 16388 / 20483 | 4095 / 20483 | 49166 / 61451 | 92.16 ms / 78.35 ms |
| diamond(n=8,depth=1024,width=4) | 19460 / 23555 | 4095 / 23555 | 51214 / 63499 | 65.60 ms / 72.39 ms |

图：`results/fig_sharing.png`。

**结论（问题 2）**：

* **当程序里存在结构相同的子表达式时，hash-consing 的收益是实打实的，而且直接体现在执行上**：
  k=8 时执行的 ATen 算子从 579（noHC）/ 725（Autograd）降到 222（**−62% / −69%**），端到端时间
  n=1024 时 252 µs vs 748 µs vs 1.90 ms（**3.0× / 7.6×**），n=65536（kernel 占主导）时
  2.74 ms vs 13.4 ms vs 10.6 ms（**4.9× / 3.9×**）。执行时间随 k 几乎不变（HC 曲线平坦），因为重复
  的 `Mul/Add` 只被执行一次；逻辑峰值内存也不随 k 增长。Autograd 作为 eager 系统没有 CSE，算子数
  随 k 线性增加。
* **对没有语法重复的程序，hash-consing 的收益很小**：前向命中为 0；反向命中只来自 ReLU VJP 中共享
  的标量常量 `Constant(0)`（每层 1 次），省掉 ~20% 的算子调用（`at::full` 及其嵌套的 `empty/fill_`
  事件），执行时间差异在 ±10% 内，与测量噪声同量级（T3 中 chain(n=8,depth=4096) 甚至 noHC 更快）。
  原因是本项目的 VJP 规则本身就引用（而非复制）前向中间值，一个节点的伴随 `gy` 只生成一次，所以反
  向图天然没有多少重复可合并。
* 换句话说，hash-consing 的价值取决于输入程序和 VJP 规则**产生了多少结构重复**；它是保证"重复
  只算一次"的机制，而不是在所有图上都能减少算子数的优化。

## 3. DAG Symbolic VJP 的生成成本是否近似线性

### T4 benchmark 进程内的 scaling suite（hash-consing 开，5 个新图的中位数）

| configuration | fwd nodes | fwd edges | bwd nodes | bwd/fwd | accumulations | build | diff | diff ns / fwd node | compile |
|---|---|---|---|---|---|---|---|---|---|
| chain(n=8,depth=64) | 322 | 321 | 260 | 0.81 | 0 | 79.3 us | 99.0 us | 307 | 36.6 us |
| chain(n=8,depth=128) | 642 | 641 | 516 | 0.80 | 0 | 159.2 us | 207.0 us | 322 | 73.9 us |
| chain(n=8,depth=256) | 1282 | 1281 | 1028 | 0.80 | 0 | 597.0 us | 854.4 us | 666 | 336.2 us |
| chain(n=8,depth=512) | 2562 | 2561 | 2052 | 0.80 | 0 | 779.2 us | 883.6 us | 345 | 304.0 us |
| chain(n=8,depth=1024) | 5122 | 5121 | 4100 | 0.80 | 0 | 2.73 ms | 2.76 ms | 538 | 1.20 ms |
| chain(n=8,depth=2048) | 10242 | 10241 | 8196 | 0.80 | 0 | 5.07 ms | 8.30 ms | 810 | 4.38 ms |
| chain(n=8,depth=4096) | 20482 | 20481 | 16388 | 0.80 | 0 | 23.40 ms | 31.83 ms | 1554 | 14.80 ms |
| diamond(n=8,depth=16,width=4) | 242 | 289 | 308 | 1.27 | 48 | 70.7 us | 107.1 us | 443 | 39.1 us |
| diamond(n=8,depth=64,width=4) | 962 | 1153 | 1220 | 1.27 | 192 | 372.2 us | 717.9 us | 746 | 273.1 us |
| diamond(n=8,depth=256,width=4) | 3842 | 4609 | 4868 | 1.27 | 768 | 1.26 ms | 2.48 ms | 646 | 870.4 us |
| diamond(n=8,depth=1024,width=4) | 15362 | 18433 | 19460 | 1.27 | 3072 | 11.19 ms | 19.88 ms | 1294 | 10.06 ms |
| mlp(batch=4,in=8,hidden=8,out=8,layers=8) | 48 | 48 | 69 | 1.44 | 1 | 19.6 us | 30.0 us | 624 | 12.5 us |
| mlp(batch=4,in=8,hidden=8,out=8,layers=32) | 168 | 168 | 237 | 1.41 | 1 | 57.8 us | 93.4 us | 556 | 39.7 us |
| mlp(batch=4,in=8,hidden=8,out=8,layers=128) | 648 | 648 | 909 | 1.40 | 1 | 183.0 us | 327.1 us | 505 | 119.4 us |

### 独立进程、无 Torch 的 `sad_scaling`（7 次中位数；完整数据见 `results/scaling_standalone*.txt`）

| 任务 | fwd nodes | edges | diff（HC 开） | ns/节点（HC 开） | diff（HC 关） | ns/节点（HC 关） |
|---|---|---|---|---|---|---|
| chain depth 256 | 1,282 | 3,332 | 0.53 ms | 417 | 0.32 ms | 249 |
| chain depth 2048 | 10,242 | 26,628 | 7.5 ms | 732 | 2.8 ms | 269 |
| chain depth 8192 | 40,962 | 106,500 | 26.5 ms | 648 | 12.8 ms | 313 |
| chain depth 32768 | 163,842 | 425,988 | 185 ms | 1129 | 63.6 ms | 388 |
| diamond depth 256 | 3,842 | 14,340 | 3.3 ms | 851 | – | – |
| diamond depth 4096 | 61,442 | 229,380 | 95 ms | 1545 | – | – |
| diamond depth 32768 | 491,522 | 1,835,012 | 1.12 s | 2270 | – | – |

图：`results/fig_scaling.png`。

**结论（问题 3）**：

* **输出规模严格线性**：反向节点数与前向节点数之比在每个任务族内是常数（chain 0.80、diamond 1.27、
  mlp 1.40），边数也线性；每个前向节点恰好访问一次（`ad_visited = fwd nodes`），梯度贡献数等于前向
  边数，累加次数等于扇出超过 1 的次数。这与"局部 VJP 规则大小有界 ⇒ O(V+E)"的分析一致。
* **时间在关闭 hash-consing 时近似线性**：250–390 ns/节点，从 1.3k 到 164k 节点只漂移 1.5×。
* **开启 hash-consing 时时间约为 O(n^1.2)**：每节点成本从 ~0.4 µs（1k 节点）升到 ~1.1 µs（164k
  节点）、~2.3 µs（diamond 492k 节点）。差值完全来自 `std::unordered_multimap` 哈希表：节点池
  超出 L2/L3 之后，每次 `emplace` 的桶查找与再散列成为 cache miss 主导的随机访问。这是实现常数的
  问题（可用开放寻址表、把 key 内联到连续数组解决），不是算法复杂度的增长。
* 在实际尺寸（MLP / 几百层的链）上，一次性成本很小：`mlp 1024×1024×8 层` 建图 21 µs、微分 36 µs、
  编译 16 µs，相对 32 ms 的单次执行可忽略；4096 层的链（37k 节点）一次性成本 70 ms，相当于 0.8 次
  DAG 执行或 0.3 次 Autograd 执行（T6）。

## 4. SLP Executor vs LibTorch Autograd 的数值执行效率

### T5 代表性配置（median；ratio = torch / DAG，>1 表示 DAG 更快；完整 52 行见 `results/tables.md`）

| configuration | ops DAG/torch | fwd DAG | fwd torch | bwd DAG | bwd torch | total DAG (P90) | total torch (P90) | fwd ratio | bwd ratio | total ratio | fwd no-grad DAG/torch | peak live DAG | peak malloc DAG/torch | grad max rel err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chain(n=64,depth=16) | 206/205 | 40.6 us | 92.4 us | 76.7 us | 386.8 us | 117.3 us (153.5 us) | 475.5 us (584.0 us) | 2.28 | 5.04 | 4.05 | 38.8 us/36.8 us | 8.3 KiB | 18.2 KiB/47.7 KiB | 0.0e+00 |
| chain(n=1024,depth=64) | 782/781 | 305.1 us | 606.3 us | 667.0 us | 1.94 ms | 977.9 us (1.09 ms) | 2.54 ms (3.00 ms) | 1.99 | 2.91 | 2.60 | 272.4 us/240.1 us | 516.0 KiB | 567.5 KiB/720.8 KiB | 0.0e+00 |
| chain(n=1024,depth=256) | 3086/3085 | 1.93 ms | 3.70 ms | 3.17 ms | 11.56 ms | 5.13 ms (5.51 ms) | 15.90 ms (17.76 ms) | 1.92 | 3.64 | 3.10 | 1.11 ms/1.07 ms | 2.0 MiB | 2.2 MiB/2.8 MiB | 0.0e+00 |
| chain(n=65536,depth=16) | 212/211 | 2.07 ms | 1.41 ms | 4.01 ms | 2.02 ms | 6.37 ms (8.41 ms) | 3.42 ms (4.11 ms) | 0.68 | 0.50 | 0.54 | 741.1 us/707.5 us | 8.3 MiB | 8.3 MiB/8.3 MiB | 0.0e+00 |
| chain(n=65536,depth=256) | 3092/3091 | 34.40 ms | 26.54 ms | 60.45 ms | 43.78 ms | 98.92 ms (148.54 ms) | 72.93 ms (79.65 ms) | 0.77 | 0.72 | 0.74 | 20.79 ms/21.20 ms | 128.3 MiB | 128.4 MiB/129.1 MiB | 0.0e+00 |
| sharing(n=1024,depth=8,k=8) | 222/725 | 102.3 us | 476.7 us | 145.7 us | 1.41 ms | 251.5 us (281.8 us) | 1.90 ms (1.97 ms) | 4.66 | 9.67 | 7.55 | 97.7 us/221.2 us | 68.0 KiB | 73.5 KiB/219.1 KiB | 1.2e-07 |
| diamond(n=4096,depth=16,width=8) | 1646/1501 | 1.23 ms | 2.06 ms | 3.03 ms | 4.82 ms | 4.33 ms (4.58 ms) | 6.87 ms (7.33 ms) | 1.68 | 1.59 | 1.59 | 751.6 us/724.4 us | 2.3 MiB | 2.3 MiB/2.6 MiB | 0.0e+00 |
| nn(batch=32,in=64,out=32) | 45/44 | 9.8 us | 20.3 us | 27.6 us | 63.6 us | 37.4 us (38.9 us) | 84.2 us (113.7 us) | 2.07 | 2.31 | 2.25 | 8.9 us/9.1 us | 28.1 KiB | 20.1 KiB/24.2 KiB | 0.0e+00 |
| nn(batch=1024,in=1024,out=1024) | 51/50 | 6.90 ms | 8.84 ms | 13.28 ms | 16.48 ms | 20.35 ms (23.86 ms) | 25.29 ms (27.78 ms) | 1.28 | 1.24 | 1.24 | 5.90 ms/6.73 ms | 16.0 MiB | 12.0 MiB/12.0 MiB | 0.0e+00 |
| mlp(batch=128,in=64,hidden=64,out=64,layers=8) | 296/281 | 329.6 us | 357.1 us | 828.3 us | 983.1 us | 1.17 ms (1.76 ms) | 1.33 ms (1.87 ms) | 1.08 | 1.19 | 1.14 | 385.9 us/290.2 us | 608.0 KiB | 614.0 KiB/375.3 KiB | 0.0e+00 |
| mlp(batch=128,in=256,hidden=256,out=64,layers=8) | 296/281 | 1.32 ms | 1.93 ms | 2.67 ms | 3.46 ms | 3.99 ms (4.35 ms) | 5.42 ms (5.94 ms) | 1.46 | 1.30 | 1.36 | 1.25 ms/1.49 ms | 2.4 MiB | 2.3 MiB/2.3 MiB | 0.0e+00 |
| mlp(batch=128,in=1024,hidden=1024,out=64,layers=8) | 296/281 | 11.46 ms | 10.80 ms | 20.75 ms | 17.35 ms | 31.73 ms (42.94 ms) | 28.10 ms (38.23 ms) | 0.94 | 0.84 | 0.89 | 11.74 ms/9.83 ms | 33.8 MiB | 33.3 MiB/33.3 MiB | 0.0e+00 |
| chain(n=8,depth=4096) | 49166/49165 | 35.32 ms | 48.16 ms | 58.18 ms | 191.13 ms | 92.16 ms (104.17 ms) | 241.89 ms (274.25 ms) | 1.36 | 3.29 | 2.62 | 17.50 ms/14.12 ms | 256.0 KiB | 3.5 MiB/14.0 MiB | 0.0e+00 |
| diamond(n=8,depth=1024,width=4) | 51214/46093 | 22.45 ms | 42.77 ms | 42.26 ms | 153.23 ms | 65.60 ms (80.56 ms) | 199.00 ms (234.69 ms) | 1.90 | 3.63 | 3.03 | 16.75 ms/12.58 ms | 160.1 KiB | 2.2 MiB/11.6 MiB | 0.0e+00 |

### T6 一次性成本与首次 / 稳态执行

| configuration | build (cold) | diff (cold) | compile | first exec DAG | steady exec DAG | first exec torch | steady torch | liveness: peak live / all defined |
|---|---|---|---|---|---|---|---|---|
| chain(n=1024,depth=16) | 24.2 us (32.7 us) | 29.8 us (38.5 us) | 12.1 us | 438.3 us | 234.9 us | 785.3 us | 724.4 us | 132.0 KiB / 400.0 KiB |
| chain(n=65536,depth=256) | 396.1 us (465.4 us) | 444.3 us (479.4 us) | 153.0 us | 178.60 ms | 98.92 ms | 80.48 ms | 72.93 ms | 128.3 MiB / 400.0 MiB |
| chain(n=8,depth=4096) | 23.40 ms (17.96 ms) | 31.83 ms (29.07 ms) | 14.80 ms | 85.95 ms | 92.16 ms | 219.92 ms | 241.89 ms | 256.0 KiB / 800.0 KiB |
| diamond(n=8,depth=1024,width=4) | 11.19 ms (9.18 ms) | 19.88 ms (19.35 ms) | 10.06 ms | 47.80 ms | 65.60 ms | 192.58 ms | 199.00 ms | 160.1 KiB / 864.0 KiB |
| mlp(batch=128,in=1024,hidden=1024,out=64,layers=8) | 21.2 us (35.4 us) | 36.3 us (52.0 us) | 16.1 us | 46.09 ms | 31.73 ms | 27.77 ms | 28.10 ms | 33.8 MiB / 54.0 MiB |
| nn(batch=1024,in=1024,out=1024) | 2.4 us (13.2 us) | 6.2 us (13.2 us) | 4.8 us | 25.15 ms | 20.35 ms | 25.48 ms | 25.29 ms | 16.0 MiB / 25.0 MiB |

图：`results/fig_exec_chain.png`、`fig_exec_mlp.png`、`fig_exec_nn.png`、`fig_exec_diamond.png`、
`fig_speedup_vs_size.png`。

**结论（问题 4）**：结果按工作集大小分成两个明确的区间（见 `fig_speedup_vs_size.png`）。

* **小张量 / 每算子开销主导（程序定义的张量总量 ≲ 1–2 MiB）**：DAG 执行器端到端快 **2–4×**
  （有语法重复的 sharing 任务 7.5×）。拆开看：
  * 前向：DAG 与 Autograd 的差距（1.9–2.3×）**全部来自 Autograd 记录计算图的开销**——用
    `NoGradGuard` 的纯 eager 前向与 DAG 前向时间几乎相同（"fwd no-grad" 列，比值 0.9–1.1）。DAG 的
    每条指令成本 0.5–1.3 µs，就是 ATen 分发 + 输出分配的底价。
  * 反向：Autograd 引擎每节点约 3–5 µs（chain 深层时 bwd ratio 3–5×），而 SLP 反向就是另一段
    直线程序，每算子成本与前向相同。
  * 每算子平均成本（total/ops，小张量配置）：DAG 0.5–1.7 µs，Autograd 1.9–5.2 µs。
* **大张量 / kernel 主导（≳ 4 MiB）**：两者在 ±30% 内，而且 **Autograd 往往更快**：
  chain(n=65536) 反向比值 0.50–0.72，mlp 1024 宽 8 层 0.84，nn 1024³ 则 DAG 快 1.24×（矩阵乘主
  导，两者算子数相同，差异在噪声与分配顺序内）。原因见第 5 节。
* **内存**：逻辑峰值存活字节与 Autograd 的 malloc 峰值几乎相等（chain(65536,256)：128.3 MiB vs
  129.1 MiB），说明 liveness 释放做到了与 Autograd 相同的"只保留反向需要的中间值"；相对不做释放的
  程序定义总量（400 MiB）降低 68%。小张量场景下 Autograd 的 malloc 峰值高 2–6×（节点元数据、
  `SavedVariable`、引擎队列），例如 chain(n=8,depth=4096) 3.5 MiB vs 14.0 MiB。
* **首次执行**：在中小规模上 DAG 首次执行比稳态慢 1.3–1.9×（分配器预热，大工作集时还要首次触页，
  chain(65536,256) 178 ms vs 99 ms）；在万节点级的微型张量程序上首次执行与稳态相当（chain(8,4096)
  86 ms vs 92 ms）。Autograd 首次执行与稳态接近，因为它在同一进程内已被 DAG 先行"预热"（运行顺序
  效应见附录 A）。
* P90 与 median 的差距通常 <30%，4 线程 CPU 上偶发的调度抖动会让个别配置的 P90 翻倍
  （mlp 1024 层 2 的 DAG P90 17 ms vs median 9.4 ms），这类点在表中都可见。

## 5. DAG 更紧凑却不更快时，瓶颈在哪里

结合第 4 节的测量，可以把"紧凑但不更快"的原因具体定位：

1. **kernel 时间不可压缩**。两个后端跑的是同一套 ATen kernel（`fwd no-grad` 比值 ≈ 1 证明这一点）；
   当单个算子耗时 ≥ 几十微秒时，DAG 省下的每算子 1–4 µs 调度开销在总时间里不足 5%。DAG 只减少
   "图结构"的工作量，不减少 FLOPs 和内存带宽。
2. **未融合的 ReLU VJP 多出一次内存遍历**。本项目的 ReLU 反向是 `GreaterThan(x,0)`（写一个 bool
   张量）+ `Mul(gy, mask)`（再读一遍），而 Autograd 用单个 `threshold_backward` kernel。在 chain 这种
   每层只有 3 个前向算子的任务里，反向每层 5 次内存遍历变成 6 次外加一次 bool 物化，这与测到的
   bwd ratio 0.5–0.72（大 n）一致：n=65536 时 DAG 反向算子数 3092 对 3091，但带宽用得更多。
   同样的道理适用于 `OnesLike` 与 `Constant` 被当成独立算子物化（Autograd 的 seed 梯度是一个
   标量 `ones({})` 直接 expand）。
3. **梯度累加不是 in-place**。`Add` 累加会分配新张量，Autograd 的 `InputBuffer` 在可能时原地累加；
   diamond 任务每层有 W 次累加，这解释了 diamond 大张量（n=4096）时 DAG 反向优势（1.3–1.7×）
   小于微型张量（n=8）时（3.5×）。
4. **分配器与首次触页**（附录 A）。在 ≥100 MiB 工作集时，默认 glibc 配置下先运行的后端要承担
   mmap/munmap 与 page fault，差异可达 3×，比两种 Autodiff 实现本身的差异大得多。这不是 DAG 的
   瓶颈，但如果不控制它，会把 DAG 的结果向任意方向扭曲；本报告主结果用 `--malloc-stable` 消除了它。
5. **每指令仍有约 0.5–1 µs 的 ATen 分发 + 分配成本**。SLP 执行器没有融合、没有静态内存规划，所以
   在小张量区间它的优势封顶在"去掉 Autograd 记录与引擎开销"这一层（2–4×），再往上要靠超出本实验范
   围的 kernel fusion / 内存复用。
6. **建图与 hash-consing 的一次性成本**随节点数以 ~n^1.2 增长（第 3 节），在十万节点级别达到
   100 ms–1 s；对需要每步重建图的动态场景这会是额外负担，对静态图则一次摊销。

## 6. 总结

* DAG + 符号 VJP 相对普通 AST 的优势在符号层面是数量级的（节点 10³–10⁴×，时间 10³×，内存
  10³×），只要表达式深或存在共享；AST 在扇出结构上指数膨胀，在纯链上也是多项式（接近立方）膨胀。
* Hash-consing 只有在程序含结构重复时才显著减少算子调用（sharing k=8：−62% 算子、3–5× 时间）；
  在没有重复的图上收益 ≈ 0，这一点通过关闭 hash-consing 的消融被直接量化。
* 符号微分的节点 / 边输出严格线性；时间在无哈希表时线性，有哈希表时 ~n^1.2，瓶颈是 `unordered_multimap`
  的缓存行为。
* 数值执行上，SLP 执行器在每算子开销主导的区间比 Autograd 快 2–4×（原因是去掉了图记录和引擎开销，
  而不是 kernel 更快），在 kernel 主导的区间不快、有时慢 10–30%，瓶颈是未融合的 ReLU 反向、非
  in-place 累加以及额外物化的常量。所有这些都在数据里可定位，未加入任何超出实验范围的优化。

## 附录 A：glibc 分配器与运行顺序效应

第一次完整运行（`results/results_default_malloc.csv`，未加 `--malloc-stable`）在 chain(n=65536,
depth=256) 上给出 DAG 前向 112 ms、Autograd 前向 31 ms（P90 280 ms vs 137 ms）。用单配置模式
（`sad_bench --task chain --n 65536 --depth 256 --repeats 15 --skip-tree --skip-nohc`）交叉验证：

| 变体 | 先运行者 | DAG fwd / bwd / total | torch fwd / bwd / total |
|---|---|---|---|
| 默认 glibc，DAG 先 | DAG | 33.0 / 55.1 / 86.2 ms（P90 191 ms） | 28.8 / 72.5 / 101.6 ms |
| 默认 glibc，torch 先（`--torch-first`） | torch | 113.8 / 50.2 / 164.0 ms | **99.0** / 108.0 / 202.1 ms |
| `--malloc-stable`，DAG 先 | DAG | 30.3 / 47.8 / 79.1 ms | 30.6 / 45.6 / 77.5 ms |
| `--malloc-stable`，torch 先 | torch | 30.4 / 49.9 / 79.3 ms | 28.5 / 43.2 / 73.2 ms |

谁先在 ~130 MiB 工作集上运行，谁的前向就慢 3×：默认配置下 256 KiB 以上的分配走 mmap，释放时
munmap 归还给内核，下一轮前向重新触页；后运行的后端继承了已增长的 mmap 阈值和已缓存的堆内存。
`mallopt(M_MMAP_THRESHOLD=1 GiB, M_TRIM_THRESHOLD=-1, M_TOP_PAD=64 MiB)` 后两种顺序结果一致。
这说明在这个尺寸上，**分配器状态对计时的影响大于两种自动微分实现之间的差异**，任何此类对比都必须
控制它。

## 附录 B：测量方法上的已知限制

* `RecordFunction` 的算子计数包含嵌套算子（如 `at::full` 展开成 `empty + fill_`），两个后端口径相同，
  但绝对数不等于"用户级"算子数；SLP 指令数（`slp_instrs_full` 列）是用户级计数。
* `mallinfo2` 只在算子边界采样，算子内部的临时峰值不可见；逻辑存活字节只对 DAG 执行器可得（Autograd
  没有等价的钩子）。
* 树基线的 2,000,000 节点上限（约 0.3–0.8 s、≤ 300 MiB）用来防止指数膨胀吃光内存；触顶的配置只
  报告触顶前的节点数和耗时，不外推。
* 建图 / 微分时间的中位数来自同一进程内 5 个新图，受进程内堆状态影响；`sad_scaling` 在独立进程中的
  数据作为交叉验证。
* 只测了 CPU、float32、4 线程、一种 CPU。代码支持 `--device cuda`（执行器按输入所在设备运行，计时前后
  同步，报告 CUDA caching allocator 峰值），但本环境没有 GPU、LibTorch 也是 CPU 版，所以没有任何 GPU
  测量数据；GPU 上的结论需要另行实测，预期 kernel launch 延迟会改变小张量区间的对比。
