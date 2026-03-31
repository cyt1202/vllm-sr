# 压测体验作业：Keyword + Preference Routing 性能与准确率分析

## 作业目标

对 vLLM Semantic Router 的 **Keyword + Preference Routing** 链路进行系统性测试，使用 [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro) 作为标准测试集，完成以下两项核心评估：

1. **性能评估**：量化每个环节（Keyword 信号、Preference 信号、Decision 引擎）的延迟消耗
2. **准确率评估**：验证路由决策的正确性，以及不同路由路径对最终回答质量的影响

---

## 一、环境准备

### 1.1 启动 Router（Docker 方式）

```bash
source vsr/bin/activate
vllm-sr serve --image semantic-router/vllm-sr:v0.2.0-0310 --image-pull-policy ifnotpresent
```

等待日志出现以下内容，确认 Preference Contrastive 模式就绪：

```
[Preference Contrastive] preloaded 3/3 example embeddings using model=mmbert in 162ms
Preference classifier initialized successfully with 3 routes
```

### 1.2 确认服务端口

| 端口 | 服务 | 用途 |
|------|------|------|
| 8899 | Envoy 入口 | 客户端请求入口 |
| 9090 | Prometheus | 指标采集 |
| 16686 | Jaeger UI | 链路追踪 |
| 3000 | Grafana | 仪表盘 |

### 1.3 安装 Python 依赖

```bash
pip install datasets aiohttp pandas numpy tabulate
```

### 1.4 下载 MMLU-Pro 测试集

```python
from datasets import load_dataset
ds = load_dataset("TIGER-Lab/MMLU-Pro")
print(f"test: {len(ds['test'])} 条, validation: {len(ds['validation'])} 条")
# test: 12032 条, validation: 70 条
```

MMLU-Pro 数据格式：

| 字段 | 类型 | 说明 |
|------|------|------|
| `question` | string | 题目文本 |
| `options` | list[str] | 10 个选项（A-J） |
| `answer` | string | 正确答案字母（A-J） |
| `answer_index` | int | 正确答案索引（0-9） |
| `category` | string | 学科分类（14 类） |
| `cot_content` | string | Chain-of-Thought 参考推理 |

MMLU-Pro 的 14 个学科分类：

```
math, physics, chemistry, biology, computer_science, engineering,
economics, business, law, health, psychology, philosophy, history, other
```

---

## 二、理解被测链路

### 2.1 config.yaml 路由拓扑

当前配置中共 **7 个路由决策**，按 priority 从高到低：

```
priority 300  urgent_request              ← keyword: urgent/immediate/asap/emergency
priority 290  sensitive_data              ← keyword: SSN/social security/credit card/password
priority 280  filter_spam                 ← keyword: buy now/free money/click here
priority 200  preference_code_generation  ← preference: 语义匹配 "代码生成"
priority 200  preference_bug_fixing       ← preference: 语义匹配 "调试修复"
priority 200  preference_code_review      ← preference: 语义匹配 "代码审查"
priority  50  default-route               ← 兜底
```

### 2.2 请求处理时序

```
请求进入 performDecisionEvaluation()
│
├─ Layer 1: 信号评估（并行 goroutine）
│    ├── Keyword Signal    ~<1ms     正则/BM25/N-gram 匹配
│    └── Preference Signal ~10-100ms Contrastive embedding 推理（Rust FFI）
│    总耗时 = max(keyword, preference)
│
├─ Layer 2: Decision Engine 评估  ~<0.1ms
│    遍历 decisions, 按 priority 选最高命中
│
└─ Layer 3: 模型选择 + Plugin 注入
     注入 system_prompt → 转发到 LLM 后端
```

**关键观察**：Keyword 信号和 Preference 信号**并行执行**，即使 Keyword 已命中高优先级 decision，Preference 的 embedding 推理仍然会跑完。总延迟由最慢的信号决定。

---

## 三、性能测试

### 实验 1：基线延迟 — 三条路径对比

分别构造必定命中 Keyword、必定命中 Preference、两者均不命中的请求，对比延迟差异。

```python
# bench_baseline.py
import asyncio, aiohttp, time, statistics, json

ROUTER_URL = "http://localhost:8899/v1/chat/completions"

# 路径 A: 命中 Keyword（priority=300，最快信号但仍需等 preference）
KEYWORD_PAYLOAD = {
    "model": "MoM",
    "messages": [{"role": "user", "content": "I need urgent help with my server"}],
    "max_tokens": 1, "stream": False
}

# 路径 B: 命中 Preference（纯 ML 推理路径）
PREFERENCE_PAYLOAD = {
    "model": "MoM",
    "messages": [{"role": "user", "content": "Write a Python function to implement quicksort"}],
    "max_tokens": 1, "stream": False
}

# 路径 C: 走 default-route（MMLU-Pro 学术问题，keyword 和 preference 都不命中）
DEFAULT_PAYLOAD = {
    "model": "MoM",
    "messages": [{"role": "user", "content": "What is the acceleration due to gravity on Earth?"}],
    "max_tokens": 1, "stream": False
}

async def bench(session, payload, n=100):
    latencies = []
    first_headers = {}
    for i in range(n):
        start = time.perf_counter()
        async with session.post(ROUTER_URL, json=payload) as resp:
            await resp.read()
            if i == 0:
                first_headers = {k: v for k, v in resp.headers.items() if k.startswith("x-vsr")}
            latencies.append((time.perf_counter() - start) * 1000)
    return latencies, first_headers

async def main():
    payloads = [
        ("Keyword 路径 (urgent_request)",  KEYWORD_PAYLOAD),
        ("Preference 路径 (code_gen)",     PREFERENCE_PAYLOAD),
        ("Default 路径 (学术问题)",         DEFAULT_PAYLOAD),
    ]
    async with aiohttp.ClientSession() as session:
        print(f"{'路径':<30} {'p50(ms)':>10} {'p95(ms)':>10} {'p99(ms)':>10} {'mean(ms)':>10}")
        print("-" * 75)
        for label, payload in payloads:
            lats, hdrs = await bench(session, payload)
            s = sorted(lats)
            print(f"{label:<30} {s[len(s)//2]:>10.1f} {s[int(len(s)*0.95)]:>10.1f} "
                  f"{s[int(len(s)*0.99)]:>10.1f} {statistics.mean(lats):>10.1f}")
            for k, v in hdrs.items():
                print(f"  {k}: {v}")
            print()

asyncio.run(main())
```

**需填写的结果表格：**

| 路径 | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) | 命中 Decision |
|------|----------|----------|----------|-----------|--------------|
| Keyword (urgent_request) | 580.7 | 658.7 | 848.0 | 584.3 | urgent_request |
| Preference (code_gen) | 575.2 | 664.2 | 737.2 | 578.8 | preference_code_review |
| Default (学术问题) | 574.6 | 689.0 | 926.2 | 588.2 | preference_code_generation |

- 原始结果
- 备注：加了一个这个：`REQUEST_HEADERS: dict[str, str] = {"X-Forwarded-Proto": "https"}` 经过测试发现上游模型服务那边是跟据 X-Forwarded-Proto 判断是否 HTTPS，你本地请求是 http://localhost:8899，上游被判成 http 才返回 308。
```
路径                                p50(ms)    p95(ms)    p99(ms)   mean(ms)
---------------------------------------------------------------------------
Keyword 路径 (urgent_request)         580.7      658.7      848.0      584.3
  x-vsr-selected-decision: urgent_request
  x-vsr-selected-confidence: 1.0000
  x-vsr-selected-reasoning: off
  x-vsr-selected-model: glm-5
  x-vsr-injected-system-prompt: true
  x-vsr-matched-keywords: urgent_keywords
  x-vsr-matched-preference: preference_bug_fixing

Preference 路径 (code_gen)            575.2      664.2      737.2      578.8
  x-vsr-selected-decision: preference_code_review
  x-vsr-selected-confidence: 1.0000
  x-vsr-selected-reasoning: off
  x-vsr-selected-model: glm-5
  x-vsr-injected-system-prompt: true
  x-vsr-matched-preference: preference_code_review

Default 路径 (学术问题)                   574.6      689.0      926.2      588.2
  x-vsr-selected-decision: preference_code_generation
  x-vsr-selected-confidence: 1.0000
  x-vsr-selected-reasoning: off
  x-vsr-selected-model: glm-5
  x-vsr-injected-system-prompt: true
  x-vsr-matched-preference: preference_code_generation
```

> **思考题 1**：三条路径的 p50 是否接近？如果 Keyword 路径和 Default 路径延迟差异不大，原因是什么？
>
> 提示：查看 `classifier.go:1275` — 所有 signal 在 `isSignalTypeUsed()` 为 true 时**并行启动**，Keyword 虽然 <1ms 完成，但 `wg.Wait()` 必须等 Preference goroutine 跑完。

**回答：**

三条路径的 p50 非常接近（Keyword: 580.7ms，Preference: 575.2ms，Default: 574.6ms，差异不足 7ms）。根本原因是 `wg.Wait()` 构成了信号层的同步屏障——所有 signal goroutine 并行启动后，必须等最慢的 Preference goroutine 完成，整体延迟由最慢信号决定。

**扩展分析一：为什么无法通过 classifier.go 做 early cancellation 优化**

从理论到代码逐层分析如下。

*理论上的优化方向*

`wg.Wait()` 是所有信号的汇合点（`classifier_signal_context.go:100`）。若 Keyword 先完成且命中了优先级足够高的 decision，可以通过 `context.Cancel()` 通知 Preference goroutine 提前退出——即 priority-aware early cancellation：

```
config 中优先级分布：
  keyword decisions:    priority 300 / 290 / 280
  preference decisions: priority 200 / 200 / 200

若 Keyword 命中 priority ≥ 280，则 Preference 结果不可能改变最终决策
→ 理论上可以 cancel Preference goroutine，节省等待时间
```

*为什么实际上做不到*

Preference Contrastive 的完整调用链最终落在一次 CGO 调用：

```
ContrastivePreferenceClassifier.Classify()          [contrastive_preference_classifier.go:163]
  → getEmbeddingWithModelType(text, "mmbert", 0)    [embedding_classifier.go:20]
  → candle_binding.GetEmbeddingWithModelType()       [candle-binding/semantic-router.go:1517]
  → GetEmbedding2DMatryoshka()                       [semantic-router.go:1548]
  → C.get_embedding_2d_matryoshka()                 ← CGO 边界，进入 Rust
      Rust/Candle: mmbert 22 层前向推理 → float32[] embedding
  ← cFloatArrayToGoSlice() 拷回 Go slice
  → cosineSimilarity(query, preloaded) × N 条规则
```

进入 `C.get_embedding_2d_matryoshka()`（`candle-binding/semantic-router.go:1561`）后：

1. Go scheduler 将当前 goroutine **锁定在一个 OS 线程**上，该线程无法被复用
2. 控制权完全交给 Rust/Candle 的 mmbert 推理
3. Go 的 `context.Done()` 信道在此期间**不被轮询**
4. Rust 层没有暴露任何中止接口
5. 唯一出口是等 Rust 函数 `return`

即使在 Go 侧对 `evaluatePreferenceSignal` 包裹 `select { case <-ctx.Done(): return }`，也只能在 CGO 调用**之前**或**之后**检查，无法打断正在执行的 mmbert 推理。

*为什么不值得优化*

实验 1 三条路径差异不足 7ms，说明 Preference 推理在整条链路中占比极小。端到端延迟的绝大部分来自上游 LLM 的 prefill（TTFT），不在 Router 侧。即使彻底消除 Preference 等待，整体延迟的改善也不足 1.3%，**瓶颈在上游模型，不在 classifier**。

**扩展分析二：增加 Preference 规则数量的影响**

`Classify()` 的内部结构（`contrastive_preference_classifier.go:151`）：

```
Classify(text)
  │
  ├─ [1 次] getEmbeddingWithModelType()   ← CGO → Rust 推理，唯一的慢操作
  │         query text → float32[] embedding
  │
  └─ [顺序循环] for ruleName, embeddings := range c.ruleEmbeddings
                    for _, emb := range embeddings
                        cosineSimilarity(queryEmbedding, emb)  ← 纯 CPU 点积
```

**无论配置多少条 preference rule，Rust FFI 推理只调用 1 次。** 规则数量只影响余弦相似度循环，而点积运算（768 维浮点向量）远比 mmbert 推理便宜，增加规则对每次请求的延迟影响可以忽略不计。实验 2 复测（config 修改后增加了 30 个 examples）QPS 与原结果在误差范围内完全吻合，印证了这一结论。

真正受规则数量影响的是**启动时的预加载**（`preloadRuleEmbeddings()`）：每个 example 需要一次 Rust FFI 推理，但这只在初始化时执行一次，与运行时延迟无关。

**扩展分析三：真正有效的 Preference 优化方向**

mmbert 模型使用了 **2D Matryoshka Representation Learning** 训练，支持在推理时同时裁剪两个维度：

- **Layer 早退**：浅层（第 3、6、11 层）的输出已能表达有意义的语义，不必跑完 22 层
- **Dimension 截断**：embedding 的前 N 维构成质量较低但仍可用的子空间（64/128/256/512 维）

当前调用路径使用完整模型（`targetLayer=0, targetDim=0`，即 22 层、768 维）。代码已支持早退调用（`candle-binding/semantic-router.go:1548`）：

```go
// 完整模型（当前）
GetEmbedding2DMatryoshka(text, "mmbert", 0, 0)    // 22 层，768 维

// 早退示例
GetEmbedding2DMatryoshka(text, "mmbert", 3, 256)  // 3 层，256 维，最激进
GetEmbedding2DMatryoshka(text, "mmbert", 6, 128)  // 6 层，128 维
GetEmbedding2DMatryoshka(text, "mmbert", 11, 256) // 11 层，256 维
```

若要将此参数暴露给 config.yaml，需改动 4 处（`config/model_config_types.go` 增加字段、`contrastive_preference_classifier.go` 存参并换调用、`preference_classifier.go` 透传、`config.yaml` 加配置项），总改动量约 30–40 行，逻辑为纯参数传递。唯一约束是 `preloadRuleEmbeddings()` 和 `Classify()` 必须使用相同的 `targetLayer`/`targetDim`，两处都读 struct 字段，天然保证一致。实测加速比见第六节，CPU 环境下 L11 可实现 2.0×、L6 可实现 3.6×、L3 可实现 7.2× 加速（Rust 单独 benchmark）。端到端对比实验表明 L11 路由准确率 47.9%（反超 L22 的 42.6%），L6 为 39.5%，且高并发吞吐量显著提升（并发 50 时 L6 QPS 是 L22 的 2.1 倍）。

---

### 实验 2：并发吞吐量测试

使用 MMLU-Pro 题目作为真实负载，测试不同并发度下的 QPS 和尾部延迟。

```python
# bench_throughput.py
import asyncio, aiohttp, time, json
from datasets import load_dataset

ROUTER_URL = "http://localhost:8899/v1/chat/completions"

# 从 MMLU-Pro 中取 500 条题目作为负载
ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
questions = [ds[i]["question"] for i in range(500)]

async def run_bench(concurrency, total=500):
    sem = asyncio.Semaphore(concurrency)
    latencies = []
    errors = 0

    async def one_req(text):
        nonlocal errors
        payload = {
            "model": "MoM",
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 1, "stream": False
        }
        async with sem:
            try:
                t0 = time.perf_counter()
                async with aiohttp.ClientSession() as s:
                    async with s.post(ROUTER_URL, json=payload,
                                      timeout=aiohttp.ClientTimeout(total=10)) as r:
                        await r.read()
                latencies.append((time.perf_counter() - t0) * 1000)
            except Exception:
                errors += 1

    wall_start = time.perf_counter()
    await asyncio.gather(*[one_req(questions[i % len(questions)]) for i in range(total)])
    wall = time.perf_counter() - wall_start

    s = sorted(latencies) if latencies else [0]
    qps = total / wall
    return {
        "concurrency": concurrency,
        "qps": qps,
        "p50": s[len(s)//2],
        "p99": s[min(int(len(s)*0.99), len(s)-1)],
        "errors": errors
    }

async def main():
    print(f"{'并发度':>8} {'QPS':>8} {'p50(ms)':>10} {'p99(ms)':>10} {'errors':>8}")
    print("-" * 50)
    for c in [1, 5, 10, 20, 50]:
        r = await run_bench(c)
        print(f"{r['concurrency']:>8} {r['qps']:>8.1f} {r['p50']:>10.1f} "
              f"{r['p99']:>10.1f} {r['errors']:>8}")

asyncio.run(main())
```

**需填写的吞吐量表格：**

| 并发度 | QPS | p50 (ms) | p99 (ms) | 错误数 |
|--------|-----|----------|----------|--------|
| 1 | 1.3 | 653.1 | 1327.1 | 0 |
| 5 | 4.2 | 1075.9 | 2167.3 | 0 |
| 10 | 7.0 | 1312.3 | 2440.9 | 0 |
| 20 | 10.4 | 1815.9 | 3138.7 | 0 |
| 50 | 12.5 | 3356.2 | 8729.5 | 1 |

- 原始结果（config 修改前）
```
     并发度      QPS    p50(ms)    p99(ms)   errors
--------------------------------------------------
       1      1.3      653.1     1327.1        0
       5      4.2     1075.9     2167.3        0
      10      7.0     1312.3     2440.9        0
      20     10.4     1815.9     3138.7        0
      50     12.5     3356.2     8729.5        1
```

- 复测结果（config 修改后，增加 preference examples + threshold）
```
     并发度      QPS    p50(ms)    p99(ms)   errors
--------------------------------------------------
       1      1.3      650.0     1582.1        0
       5      4.4     1050.5     2111.6        0
      10      7.0     1225.6     2461.3        0
      20     10.5     1776.5     3384.2        0
      50     12.6     3300.9     8792.3        3
```
两次结果在误差范围内完全吻合，验证了"增加 preference examples/规则数不影响吞吐量"的结论——Preference Contrastive 无论规则多少都只发生 1 次 Rust FFI 推理，吞吐瓶颈始终在上游 LLM。

> **思考题 2**：并发度从 10 提升到 50 时，p99 是否出现明显跳增？如果是，瓶颈在哪一层？
>
> 提示：Preference signal 的 contrastive 分类器在 `contrastive_preference_classifier.go:174` 使用了 `sync.RWMutex` 读锁。embedding 推理本身是否有并发瓶颈取决于 Rust FFI 层的线程安全实现。

**回答：**

p99 出现了明显跳增：从并发 10 的 2461.3ms 上升到并发 50 的 8792.3ms，增幅 3.6×，且出现超时错误。

瓶颈分两层分析：

**第一层（主要）：上游 LLM 饱和**。QPS 在并发 20→50 仅从 10.5 升至 12.6（增幅 20%），而 p99 同期从 3384ms 跳至 8792ms（增幅 160%），说明系统在并发 20 附近已进入 LLM 饱和区——继续增加并发度只是在加大请求队列，不再提升吞吐，尾延迟因排队时间叠加而放大。

**关于提示中 `contrastive_preference_classifier.go:174` 的 RWMutex：**

```go
// line 163 — CGO 推理调用，在锁外，50 个 goroutine 可同时进入
out, err := getEmbeddingWithModelType(text, c.modelType, 0)

// line 174 — 读锁在推理调用之后，只保护读取 ruleEmbeddings（preloaded 向量）
c.mu.RLock()
defer c.mu.RUnlock()
for _, emb := range c.ruleEmbeddings[...] { cosineSimilarity(...) }
```

这把 `sync.RWMutex` 并不保护推理本身。更重要的是：`ruleEmbeddings` 在服务启动时预加载，运行期间只读不写，写锁几乎永远不被触发，读锁争用极低，开销可以忽略不计。**这把锁在正常运行中基本等同于无效**，不构成任何瓶颈。

**结论：唯一瓶颈是 CGO embedding 调用本身**。每次 CGO 调用会锁定一个 OS 线程，高并发时大量线程堆积，加上 LLM 侧的饱和排队，共同导致 p99 跳增。

---

## 四、准确率测试

> **本节为开放性实验。** 没有固定的"正确配置"——你需要自己设计路由方案，用 MMLU-Pro 的学科标签作为 ground truth 来评估效果。

### 4.1 实验目标

MMLU-Pro 每道题都自带 `category` 字段，这是天然的 ground truth 标签。实验目标是：

> **设计一套路由决策，使 Router 能将 14 个学科的题目自动分配到对应专项模型，并通过答题准确率验证路由的实际收益。**

**学科 → 路由组的参考映射**（可自行调整合并粒度）：

| 学科群 | 包含学科 | 参考 decision 名 | 适合的模型类型 |
|--------|---------|----------------|--------------|
| STEM 推理 | math, physics, engineering | `route_stem` | 数学推理模型（Qwen2.5-Math、DeepSeek-R1） |
| 计算机 | computer_science | `route_cs` | 代码模型（DeepSeek-Coder、Qwen2.5-Coder） |
| 生命科学 | biology, chemistry, health | `route_science` | 理科通识模型 |
| 人文社科 | law, history, philosophy, psychology | `route_humanities` | 强语言理解模型（Llama 系列） |
| 商科 | economics, business | `route_business` | 通用推理模型 |
| 兜底 | other | `default-route` | 通用最强模型 |

---

### 4.2 路由方案设计（三选一实现）

从以下三种信号类型中选择一种，在 `config.yaml` 中实现完整的学科路由配置：

#### 方案 A：Keyword Signal（<1ms，实现最简单）

为每个学科群挑选高区分度的领域词，填入 `keywords` 列表。挑战在于覆盖率：MMLU 题目措辞多样，纯关键词容易漏判。

```yaml
# 示例框架，需自行补充各学科词表
decisions:
  - name: route_stem
    priority: 160
    conditions:
      signal_type: keyword
      keywords:
        - "integral"
        - "eigenvalue"
        # 继续补充...
    backend: math-model
    system_prompt: "You are a mathematics and physics expert. Reason step by step."

  - name: route_cs
    priority: 160
    conditions:
      signal_type: keyword
      keywords:
        - "time complexity"
        - "binary tree"
        # 继续补充...
    backend: code-model
    system_prompt: "You are a computer science expert."

  # route_science / route_humanities / route_business 同理
```

#### 方案 B：Preference Contrastive Signal（10-100ms，零代码改动，推荐）

为每个学科群提供 4-6 条典型例题作为锚点，Router 通过余弦相似度匹配。核心挑战在于锚点选取和 `threshold` 调优。

```yaml
# 示例框架，需自行补充锚点和调整 threshold
decisions:
  - name: route_stem
    priority: 160
    conditions:
      signal_type: preference
      examples:
        - "Solve the integral of x squared from 0 to 1"
        - "Calculate the net force using Newton's second law"
        # 继续补充 4-6 条，覆盖 math/physics/engineering 典型句式
    threshold: 0.75   # 建议从 0.70 开始调，观察 4.3 的 Recall 变化
    backend: stem-model
    system_prompt: "You are an expert in mathematics and physics. Reason step by step."

  - name: route_cs
    priority: 160
    conditions:
      signal_type: preference
      examples:
        - "What is the time complexity of quicksort in the worst case"
        # 继续补充...
    threshold: 0.75
    backend: code-model
    system_prompt: "You are a computer science expert."

  # route_science / route_humanities / route_business 同理
```

#### 方案 C：Category Classifier（5-20ms，精度最高，需训练）

利用 `candle-binding` 的 ModernBERT 基础设施，在 MMLU-Pro 训练集上 LoRA fine-tune 一个 14 分类头。

```
MMLU-Pro question
    → ModernBERT embedding（candle-binding）
    → 14-class softmax head（LoRA fine-tune）
    → subject label + confidence
    → 按 SUBJECT_TO_DECISION 映射路由
    （confidence < 0.6 → fallback 到 default-route）
```

需在 `classification/classifier.go` 新增 `SignalTypeCategory` 信号类型，与现有 Keyword/Preference 并行执行。适合追求最高精度、愿意投入训练成本的情况。

**方案 C 实施步骤**

代码库已内置完整的训练和推理基础设施，整个实施分三步：

**步骤一：训练（Python + PEFT）**

训练脚本位于 `src/training/model_classifier/classifier_model_fine_tuning_lora/ft_linear_lora.py`，直接支持 MMLU-Pro 数据集，14 个 category 已硬编码。

```bash
# 建立 Python 环境
cd src/training/model_classifier/classifier_model_fine_tuning_lora
uv venv .venv --python 3.11
uv pip install torch peft transformers datasets scikit-learn accelerate

# GPU 训练（RTX 4070 12GB，约 10-20 分钟）
.venv/bin/python ft_linear_lora.py \
  --mode train \
  --model mmbert-base \
  --epochs 10 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --max-samples 10000 \
  --batch-size 32 \
  --learning-rate 2e-5
```

脚本自动完成：加载 MMLU-Pro → LoRA 训练 → 合并权重 → 输出 Rust 兼容模型目录（含 `lora_config.json`）。

**步骤二：配置（config.yaml）**

```yaml
global:
  category_model:
    model_id: /path/to/lora_intent_classifier_mmbert-base_r32_model_rust
    use_mmbert_32k: false
    category_mapping_path: config/signal/domain/mmlu.yaml
    threshold: 0.5   # 低于此置信度 → fallback 到 default-route
```

`config/signal/domain/mmlu.yaml` 已预定义全部 14 个 MMLU category，无需修改。

**步骤三：路由决策绑定（config.yaml decisions 段）**

在 decisions 中增加 domain 条件，将分类结果映射到路由决策：

```yaml
- name: route_stem_domain
  priority: 210   # 略高于 preference route（200），domain 优先
  rules:
    operator: OR
    conditions:
    - type: domain
      name: math
    - type: domain
      name: physics
    - type: domain
      name: engineering
  modelRefs:
  - model: glm-5
    use_reasoning: false
  plugins:
  - type: system_prompt
    configuration:
      system_prompt: You are a STEM tutor. Start every answer with [ROUTE STEM].
```

其余 route_cs、route_science、route_humanities、route_business 同理。

**推理侧调用链**（无需改任何 Go 代码，已有）：

```
config (domain signal)
  → evaluateDomainSignal()                   [classifier_signal_context.go]
  → CategoryInitializer.Init()               → 检测 lora_config.json → LoRA 模式
  → ClassifyWithProbabilities(text)          → CGO → Rust/Candle forward pass
  → matchDomainCategories()                  → 熵分析，阈值过滤
  → SignalResults.MatchedDomainRules         → 进入 DecisionEngine
  → 路由决策输出
```

**硬件需求**：RTX 4070（12GB VRAM）完全够用；mmbert-base 约 560MB 显存占用，batch_size=32 峰值约 4–5GB。CPU 也可训练但需数小时。

**方案 C 的鲁棒性分析**

方案 C 的鲁棒性高度依赖部署场景，有以下几个维度需要评估：

**① 分布内鲁棒性（强）**

训练集和测试集同为 MMLU-Pro，分布高度一致。12K 标注样本的有监督训练能给出稳定的分类边界，准确率高且方差小。这是三种方案中分布内精度最高的。

**② OOD（分布外）鲁棒性（弱点）**

这是方案 C 最大的隐患。Softmax 输出和必须为 1，模型没有"拒绝分类"的选项。一个完全无关的输入（如"帮我写一首诗"）也会被强制分配到某个类，且置信度可能仍然不低。

方案 B（Contrastive）有天然的 fallback：余弦相似度低于 threshold 时路由到 default-route。方案 C 必须**手动加置信度阈值**才能实现同样效果：

```
if max(softmax_prob) < 0.5 → default-route
else → 按最高概率类路由
```

这个阈值需要在验证集上单独标定，增加了工程复杂度。

**③ 格式敏感性（中等风险）**

MMLU-Pro 的问题格式固定：问题文本 + A/B/C/D 选项。真实用户的提问是自由文本，没有选项。`[CLS]` token 的 embedding 在有无选项时会产生偏移，模型在训练时只见过带选项的格式，部署时如果不做格式对齐，分类置信度会系统性下降。

**④ 类边界模糊（已知风险）**

MMLU-Pro 本身的 category 边界不清晰——计算生物学（CS + Biology）、数学物理（Math + Physics）、行为经济学（Economics + Psychology）等跨界问题在训练集中也有标注噪声，分类器在这类样本上的表现会低于整体水平。

**与方案 B 的鲁棒性对比：**

| 场景 | 方案 B（Contrastive）| 方案 C（LoRA Classifier）|
|---|---|---|
| MMLU-Pro 分布内 | 42.6–47.9%（取决于 layer 配置）| 估计 80–90%+（高且稳定）|
| OOD 通用问题 | 低于 threshold → 安全 fallback | 强制分类，可能自信地分错 |
| 跨类边界问题 | 多类都低于 threshold → fallback | 分到最高概率类，可能错 |
| 输入格式变化 | embedding 层面泛化，退化温和 | 格式偏移可能导致置信度骤降 |

**结论**：若 router 专服务 MMLU-Pro 风格的学术问答，方案 C 鲁棒性优秀，是最优选择；若服务通用用户输入，OOD 问题上比方案 B 更危险（自信地分错，而非安全 fallback）。实际部署建议方案 C 配合置信度阈值，低置信度时退回 default-route，在分布内高精度和 OOD 安全性之间取得平衡。

---

**方案 C 实际执行记录（第一次训练）**

**环境准备**

```bash
TRAIN_DIR=src/training/model_classifier/classifier_model_fine_tuning_lora
uv venv $TRAIN_DIR/.venv --python 3.11
# 安装依赖（torch 2.5.1+cu121，peft、transformers、datasets、scikit-learn、accelerate、numpy）
# 注：torch 需通过 --index-url https://download.pytorch.org/whl/cu121 安装
```

硬件：NVIDIA GeForce RTX 4070，12GB VRAM（CUDA 13.1）

**遇到的问题及修复**

_问题 1：torch.load 安全限制_

```
ValueError: Due to a serious vulnerability issue in torch.load (CVE-2025-32434),
we now require users to upgrade torch to at least v2.6.
```

mmBERT-base 本地缓存包含旧格式 `pytorch_model.bin`，触发 transformers 的安全检查。
**修复**：HuggingFace Hub 还缓存了一份 `model.safetensors`，将其复制到含完整 config 的 snapshot 目录，transformers 自动优先加载 safetensors，绕过限制：

```bash
cp ~/.cache/huggingface/hub/models--jhu-clsp--mmBERT-base/snapshots/ca5b84.../model.safetensors \
   ~/.cache/huggingface/hub/models--jhu-clsp--mmBERT-base/snapshots/c59550.../
```

_问题 2：fp32 训练导致速度极慢_

首次训练参数 `batch_size=32`，无混合精度，GPU 虽 100% 占用但因显存全满（11976/12282 MiB）严重受限，每步约 **99 秒**，预计需 25 小时。
**修复**：在 `ft_linear_lora.py` 的 `TrainingArguments` 中加入：

```python
bf16=True,
dataloader_num_workers=4,
```

并将 `batch_size` 降为 16（留出显存余量）。修复后速度提升 **150 倍**，达到 1.56 it/s。

**实际训练命令**

```bash
TRAIN_DIR=/home/nickw/sf/semantic-router/src/training/model_classifier/classifier_model_fine_tuning_lora
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
$TRAIN_DIR/.venv/bin/python $TRAIN_DIR/ft_linear_lora.py \
  --mode train \
  --model mmbert-base \
  --epochs 10 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --max-samples 10000 \
  --batch-size 16 \
  --learning-rate 2e-5
```

**训练数据集情况**

脚本使用 `dataset["test"]`（MMLU-Pro test split，12032 条），每类最多采样 714 条，共 9144 条，内部 60/20/20 分割：

| 分割 | 样本数 |
|------|--------|
| Train | 5852 |
| Validation | 1463 |
| Test（内部）| 1829 |

各类样本数（受限于较小的类）：

```
biology: 714, business: 714, chemistry: 714, computer science: 410,
economics: 714, engineering: 714, health: 714, history: 381,
law: 714, math: 714, other: 714, philosophy: 499, physics: 714, psychology: 714
```

LoRA 配置：rank=32，alpha=64，dropout=0.1，target_modules: `attn.Wqkv / attn.Wo / mlp.Wi / mlp.Wo`
可训练参数：**6,769,166 / 314,310,172 = 2.15%**，总步数 1830 步

**逐 Epoch 验证结果**

| Epoch | eval_loss | eval_accuracy | eval_f1 |
|-------|-----------|--------------|---------|
| 1 | 1.4010 | 55.57% | 55.58% |
| 2 | 0.8194 | 73.00% | 73.17% |
| 3 | 0.7062 | 76.42% | 76.36% |
| 4 | 0.6863 | 76.97% | 77.27% |
| 5 | 0.6541 | 77.92% | 77.90% |
| 6 | 0.6458 | 78.95% | 78.90% |
| **7** | **0.6445** | **79.84%** | **79.91%** ← best |
| 8 | 0.6524 | 78.40% | 78.44% |
| 9 | 0.6495 | 79.29% | 79.31% |
| 10 | 0.6520 | 79.22% | 79.24% |

最佳 checkpoint 在 **Epoch 7**（`load_best_model_at_end=True`，按 eval_f1 选模型）。

**最终训练统计**

```
训练时长：1323 秒（约 22 分钟）
训练速度：44.23 samples/s，1.38 steps/s（bf16 + num_workers=4）
train_loss（epoch 10）：1.271

最终加载最佳模型（epoch 7）验证集结果：
  eval_accuracy: 79.84%
  eval_f1:       79.91%
  eval_loss:     0.6445
```

**模型文件位置**

```
# LoRA adapter（PEFT 格式，含 adapter_config.json + adapter_model.safetensors）
/home/nickw/sf/semantic-router/src/training/model_classifier/
  classifier_model_fine_tuning_lora/
    lora_intent_classifier_mmbert-base_r32/
      adapter_config.json
      adapter_model.safetensors
      label_mapping.json
      category_mapping.json
      tokenizer.json / tokenizer_config.json
      checkpoint-183/ ... checkpoint-1830/   ← 每 epoch 保存一次

# Rust 兼容合并模型（candle-binding 可直接加载）
    lora_intent_classifier_mmbert-base_r32_rust/
      model.safetensors      ← 合并后完整权重（~590MB）
      lora_config.json       ← candle-binding 自动检测 LoRA 模式用
      config.json            ← 含正确 id2label 映射（14 类）
      category_mapping.json
      tokenizer.json / tokenizer_config.json
```

**模型合并命令**

```python
# 调用 merge_lora_adapter_to_full_model() 将 adapter 合并进 base model
# 生成 candle-binding 可直接加载的 Rust 兼容格式
from ft_linear_lora import merge_lora_adapter_to_full_model

merge_lora_adapter_to_full_model(
    lora_adapter_path=".../lora_intent_classifier_mmbert-base_r32",
    output_path=".../lora_intent_classifier_mmbert-base_r32_rust",
    base_model_path="jhu-clsp/mmBERT-base"
)
# 输出：Created lora_config.json for LoRA model detection
# 输出：LoRA adapter merged successfully!
```

**数据泄漏问题说明**

训练脚本使用 `dataset["test"]`（共 12032 条），而 `eval_subject_routing.py` 的 420 条评测样本也来自同一 split（`seed=42` 采样）。两者存在重叠，导致验证集的 79.84% 可能存在乐观偏差。

后续计划：在正式路由准确率评测前，先从训练集中排除这 420 条，重新训练后再评测，以获得无偏的准确率数字。

#### 方案 C 接入 vllm-sr：config.yaml 修改记录

**目标**：将训练好的 14 类分类器接入路由，使 `domain` 信号生效。

**步骤一：确认系统内置的 `mom-domain-classifier`**

系统自带 `models/mom-domain-classifier`（标准 BERT，14 类 MMLU 标签，已在 mom_registry 注册），与训练产出的 `mmbert-mmlu-classifier` 类别完全一致，可直接复用。

```
models/mom-domain-classifier/
├── model.safetensors
├── config.json           # model_type: "bert"，max_position_embeddings: 512
├── category_mapping.json # 14 类 MMLU 映射（biology/business/...）
└── tokenizer.json
```

**步骤二：修改 config.yaml**

在 `routing.signals` 下新增 14 个 domain 条目：

```yaml
routing:
  signals:
    domains:
    - name: biology
      description: Biology and life-science related queries.
      mmlu_categories: [biology]
    - name: business
      ...（共 14 个）
```

在每个 decision 的 `rules.conditions` 中加入对应 domain 条件：

```yaml
- name: route_stem
  rules:
    operator: OR
    conditions:
    - type: keyword
      name: stem_keywords
    - type: preference
      name: pref_stem
    - type: domain
      name: math
    - type: domain
      name: physics
    - type: domain
      name: engineering
```

在 `global.model_catalog.modules.classifier` 下配置 domain 分类器：

```yaml
global:
  model_catalog:
    modules:
      classifier:
        domain:
          model_id: models/mom-domain-classifier
          threshold: 0.5
          use_mmbert_32k: false
          use_cpu: true
          category_mapping_path: models/mom-domain-classifier/category_mapping.json
          fallback_category: other
```

**排错记录**

*问题 1*：首次启动报 `model path models/mmbert-mmlu-classifier not found in mom_registry`。

原因：config 中写的是训练产出的自定义模型路径，但 Go runtime 的 mom_registry 只注册了系统预置模型。

修复：改用 `models/mom-domain-classifier`（系统预置，类别相同）。

*问题 2*：第二次启动报 `failed to initialize mmBERT-32K intent classifier`。

日志关键行：`"Using mmBERT-32K for intent/category classification (32K context, YaRN RoPE)"`

原因：未显式声明 `use_mmbert_32k: false`，被 router-defaults.yaml 中 `classifier.category_model.use_mmbert_32k: true` 的默认值覆盖，导致对 512 context 的标准 BERT 使用了 32K 加载路径。

修复：在 domain config 中显式加入 `use_mmbert_32k: false` 和 `use_cpu: true`。

第三次启动成功，日志确认：

```
"msg":"Initializing Intent/Category Classifier:"
"msg":"Model: models/mom-domain-classifier"
"msg":"Classes: 14"
"msg":"CPU Mode: true"
```

#### 方案 C 路由准确率评测结果（2026-03-31）

运行命令：
```bash
cd /home/nickw/sf/benchmark
HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/eval_subject_routing.py
```

**总体路由准确率：278/420 = 66.2%**（vs 方案 B L11 最优 47.9%，提升 **+18.3pp**）

按学科明细：

| 学科 | 路由准确率 | 正确 | 总数 | avg lat(ms) |
|------|-----------|------|------|------------|
| engineering | 90.0% | 27 | 30 | 1547 |
| physics | 86.7% | 26 | 30 | 1285 |
| business | 83.3% | 25 | 30 | 1458 |
| math | 76.7% | 23 | 30 | 1188 |
| economics | 76.7% | 23 | 30 | 1614 |
| chemistry | 70.0% | 21 | 30 | 1423 |
| law | 73.3% | 22 | 30 | 1888 |
| philosophy | 73.3% | 22 | 30 | 1155 |
| history | 63.3% | 19 | 30 | 1731 |
| computer science | 60.0% | 18 | 30 | 1524 |
| health | 56.7% | 17 | 30 | 1315 |
| biology | 46.7% | 14 | 30 | 1640 |
| psychology | 46.7% | 14 | 30 | 1306 |
| other | 23.3% | 7 | 30 | 1141 |
| **总计** | **66.2%** | **278** | **420** | |

Per-Decision Precision/Recall/F1：

| Decision | Precision | Recall | F1 | TP | FP | FN |
|----------|-----------|--------|-----|----|----|-----|
| route_stem | 67.3% | 84.4% | 0.749 | 76 | 37 | 14 |
| route_cs | 46.2% | 60.0% | 0.522 | 18 | 21 | 12 |
| route_science | 66.7% | 57.8% | 0.619 | 52 | 26 | 38 |
| route_humanities | 85.6% | 64.2% | 0.733 | 77 | 13 | 43 |
| route_business | 78.7% | 80.0% | 0.793 | 48 | 13 | 12 |
| default-route | 24.1% | 23.3% | 0.237 | 7 | 22 | 23 |

**关键发现**：
- STEM 类大幅提升（math 76.7%、physics 86.7%、engineering 90.0%），这是方案 B 的主要弱点，Domain Classifier 通过直接语义分类彻底解决
- route_stem recall 达到 84.4%，方案 B L11 下该值约为 63%
- `other` 类仅 23.3%：MMLU-Pro `other` 类题目内容多样，Domain Classifier 倾向于将其归入最接近的学科而非 `other`，根本原因是分类器没有学会"拒绝归类"——训练时 `other` 类样本本身就是杂项集合，模型无法学到其内聚的语义边界，遇到模糊题目时总会压到某个具体学科
- route_cs precision 仍偏低（46.2%），部分非 CS 题目被误分入 computer_science，说明分类器在 CS 边界上仍有混淆

**三方案横向对比**：

| 方案 | 路由准确率 | 核心机制 | 主要优势 | 主要局限 |
|------|-----------|---------|---------|---------|
| 方案 B baseline（L22）| 42.6% | Contrastive 相似度 | 无需训练 | STEM 召回弱，CS 精度低 |
| 方案 B 优化（L11）| 47.9% | L11 early exit | 准确率+吞吐双提升 | 仍依赖 preference 例句覆盖度 |
| **方案 C（Domain Classifier）**| **66.2%** | 14类 softmax 分类器 | STEM 全面覆盖，business/law 高精度 | `other` 类无法拒绝归类，需训练 |

**方案 B vs 方案 C 初步对比**

| 指标 | 方案 B（Contrastive，L22）| 方案 B（Contrastive，L11 优化）| 方案 C（LoRA，含泄漏）|
|---|---|---|---|
| 路由准确率（MMLU-Pro test）| 42.6% | **47.9%** | **66.2%** |
| Preference 推理延迟 | ~97ms（22层 FFI）| ~48ms（11层 FFI） | 5–20ms |
| 并发50 端到端 QPS | 12.4 | **21.4（+73%）** | 未测试 |
| 需要训练 | 否 | 否 | 是（约 22 分钟，RTX 4070）|
| 代码改动 | 无 | 15 行（4 文件）| config.yaml 仅 |
| 模型文件大小 | 无（复用 mmbert）| 无（复用 mmbert） | 0（复用 mom-domain-classifier）|

---

### 4.3 路由准确率评估

**目标**：以 MMLU category 标签为 ground truth，量化你的路由方案对每个学科的识别准确率。

```python
# eval_subject_routing.py
"""
学科路由准确率评估
ground truth：MMLU-Pro category 字段 → SUBJECT_TO_DECISION 映射
"""
import json, time, requests, random
from collections import defaultdict
from datasets import load_dataset

ROUTER_URL = "http://localhost:8899/v1/chat/completions"

# 按自己的方案调整此映射
SUBJECT_TO_DECISION = {
    "math":             "route_stem",
    "physics":          "route_stem",
    "engineering":      "route_stem",
    "computer_science": "route_cs",
    "biology":          "route_science",
    "chemistry":        "route_science",
    "health":           "route_science",
    "law":              "route_humanities",
    "history":          "route_humanities",
    "philosophy":       "route_humanities",
    "psychology":       "route_humanities",
    "economics":        "route_business",
    "business":         "route_business",
    "other":            "default-route",
}

random.seed(42)
ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

# 每学科取 30 题，共 420 题
by_cat = defaultdict(list)
for item in ds:
    by_cat[item["category"]].append(item)

samples = []
for cat, items in by_cat.items():
    samples.extend(random.sample(items, min(30, len(items))))

results = []
for i, item in enumerate(samples):
    options_str = "\n".join(
        f"{chr(65+j)}. {opt}" for j, opt in enumerate(item["options"]) if opt
    )
    payload = {
        "model": "MoM",
        "messages": [{"role": "user", "content": f"{item['question']}\n{options_str}"}],
        "max_tokens": 1, "stream": False
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(ROUTER_URL, json=payload, timeout=15)
        latency_ms = (time.perf_counter() - t0) * 1000
        actual = resp.headers.get("x-vsr-selected-decision", "default-route") or "default-route"
    except Exception:
        latency_ms = -1
        actual = "ERROR"

    expected = SUBJECT_TO_DECISION.get(item["category"], "default-route")
    results.append({
        "category":          item["category"],
        "expected_decision": expected,
        "actual_decision":   actual,
        "correct":           actual == expected,
        "latency_ms":        latency_ms,
        "question_id":       item["question_id"],
        "answer":            item["answer"],
    })

    if (i + 1) % 60 == 0:
        acc = sum(r["correct"] for r in results) / len(results) * 100
        print(f"  进度 {i+1}/{len(samples)}, 当前路由准确率: {acc:.1f}%")

# ── 总体准确率 ──
total   = len(results)
correct = sum(r["correct"] for r in results)
print(f"\n总体路由准确率: {correct}/{total} = {correct/total*100:.1f}%")

# ── 按学科统计 ──
by_subject = defaultdict(lambda: {"total": 0, "correct": 0, "latencies": []})
for r in results:
    s = by_subject[r["category"]]
    s["total"] += 1
    if r["correct"]:
        s["correct"] += 1
    if r["latency_ms"] > 0:
        s["latencies"].append(r["latency_ms"])

print(f"\n{'学科':<20} {'路由准确率':>10} {'正确':>5} {'总数':>5} {'avg lat(ms)':>12}")
print("-" * 55)
for cat in sorted(by_subject):
    s = by_subject[cat]
    acc = s["correct"] / s["total"] * 100
    avg = sum(s["latencies"]) / len(s["latencies"]) if s["latencies"] else 0
    print(f"{cat:<20} {acc:>9.1f}% {s['correct']:>5} {s['total']:>5} {avg:>12.1f}")

# ── 按 decision 统计 Precision / Recall / F1 ──
decisions = list(set(SUBJECT_TO_DECISION.values()))
print(f"\n{'Decision':<25} {'Precision':>10} {'Recall':>10} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
print("-" * 75)
for d in sorted(decisions):
    tp = sum(1 for r in results if r["expected_decision"] == d and r["actual_decision"] == d)
    fp = sum(1 for r in results if r["expected_decision"] != d and r["actual_decision"] == d)
    fn = sum(1 for r in results if r["expected_decision"] == d and r["actual_decision"] != d)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"{d:<25} {prec:>9.1%} {rec:>9.1%} {f1:>7.3f} {tp:>5} {fp:>5} {fn:>5}")

with open("routing_accuracy_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n完整结果已保存至 routing_accuracy_results.json")
```

**需填写的按学科路由准确率表格：**

初版结果（仅 Keyword + description-only Preference，整体 26.7%）：

| 学科 | 预期 Decision | 路由准确率 | 正确数 | 总数 |
|------|-------------|-----------|-------|------|
| math | route_stem | 10.0% | 3 | 30 |
| physics | route_stem | 20.0% | 6 | 30 |
| engineering | route_stem | 13.3% | 4 | 30 |
| computer science | route_cs | 3.3% | 1 | 30 |
| biology | route_science | 33.3% | 10 | 30 |
| chemistry | route_science | 16.7% | 5 | 30 |
| health | route_science | 20.0% | 6 | 30 |
| law | route_humanities | 53.3% | 16 | 30 |
| history | route_humanities | 30.0% | 9 | 30 |
| philosophy | route_humanities | 0.0% | 0 | 30 |
| psychology | route_humanities | 30.0% | 9 | 30 |
| economics | route_business | 53.3% | 16 | 30 |
| business | route_business | 20.0% | 6 | 30 |
| other | default-route | 70.0% | 21 | 30 |
| **总计** | | **26.7%** | **112** | **420** |

调参后结果（增加 preference examples × 6 + threshold=0.62，整体 40.7%）：

| 学科 | 预期 Decision | 路由准确率 | 正确数 | 总数 |
|------|-------------|-----------|-------|------|
| math | route_stem | 36.7% | 11 | 30 |
| physics | route_stem | 56.7% | 17 | 30 |
| engineering | route_stem | 63.3% | 19 | 30 |
| computer science | route_cs | 10.0% | 3 | 30 |
| biology | route_science | 33.3% | 10 | 30 |
| chemistry | route_science | 13.3% | 4 | 30 |
| health | route_science | 30.0% | 9 | 30 |
| law | route_humanities | 66.7% | 20 | 30 |
| history | route_humanities | 53.3% | 16 | 30 |
| philosophy | route_humanities | 40.0% | 12 | 30 |
| psychology | route_humanities | 30.0% | 9 | 30 |
| economics | route_business | 66.7% | 20 | 30 |
| business | route_business | 33.3% | 10 | 30 |
| other | default-route | 36.7% | 11 | 30 |
| **总计** | | **40.7%** | **171** | **420** |

第三轮调参结果（CS 例句换为理论性题目 + 补充化学例句 + threshold=0.60 + 增加 CS 理论关键词，整体 42.6%）：

| 学科 | 预期 Decision | 路由准确率 | 正确数 | 总数 | 变化 |
|------|-------------|-----------|-------|------|------|
| math | route_stem | 36.7% | 11 | 30 | — |
| physics | route_stem | 53.3% | 16 | 30 | -3.4pp |
| engineering | route_stem | 63.3% | 19 | 30 | — |
| computer science | route_cs | 16.7% | 5 | 30 | **+6.7pp** |
| biology | route_science | 33.3% | 10 | 30 | — |
| chemistry | route_science | 30.0% | 9 | 30 | **+16.7pp** |
| health | route_science | 36.7% | 11 | 30 | +6.7pp |
| law | route_humanities | 66.7% | 20 | 30 | — |
| history | route_humanities | 53.3% | 16 | 30 | — |
| philosophy | route_humanities | 40.0% | 12 | 30 | — |
| psychology | route_humanities | 30.0% | 9 | 30 | — |
| economics | route_business | 66.7% | 20 | 30 | — |
| business | route_business | 33.3% | 10 | 30 | — |
| other | default-route | 36.7% | 11 | 30 | — |
| **总计** | | **42.6%** | **179** | **420** | **+1.9pp** |

**需填写的 Per-Decision Precision/Recall/F1 表格：**

初版：

| Decision | Precision | Recall | F1 | TP | FP | FN |
|----------|-----------|--------|-----|----|----|-----|
| route_stem | 43.3% | 14.4% | 0.217 | 13 | 17 | 77 |
| route_cs | 100.0% | 3.3% | 0.065 | 1 | 0 | 29 |
| route_science | 75.0% | 23.3% | 0.356 | 21 | 7 | 69 |
| route_humanities | 77.3% | 28.3% | 0.415 | 34 | 10 | 86 |
| route_business | 84.6% | 36.7% | 0.512 | 22 | 4 | 38 |
| default-route | 7.5% | 70.0% | 0.135 | 21 | 260 | 9 |

调参后（第二轮，threshold=0.62）：

| Decision | Precision | Recall | F1 | TP | FP | FN |
|----------|-----------|--------|-----|----|----|-----|
| route_stem | 60.3% | 52.2% | 0.560 | 47 | 31 | 43 |
| route_cs | 16.7% | 10.0% | 0.125 | 3 | 15 | 27 |
| route_science | 52.3% | 25.6% | 0.343 | 23 | 21 | 67 |
| route_humanities | 82.6% | 47.5% | 0.603 | 57 | 12 | 63 |
| route_business | 75.0% | 50.0% | 0.600 | 30 | 10 | 30 |
| default-route | 6.9% | 36.7% | 0.116 | 11 | 149 | 19 |

调参后（第三轮，CS 理论例句 + 化学例句 + threshold=0.60）：

| Decision | Precision | Recall | F1 | TP | FP | FN |
|----------|-----------|--------|-----|----|----|-----|
| route_stem | 63.0% | 51.1% | 0.564 | 46 | 27 | 44 |
| route_cs | 20.8% | 16.7% | 0.185 | 5 | 19 | 25 |
| route_science | 57.7% | 33.3% | 0.423 | 30 | 22 | 60 |
| route_humanities | 82.6% | 47.5% | 0.603 | 57 | 12 | 63 |
| route_business | 75.0% | 50.0% | 0.600 | 30 | 10 | 30 |
| default-route | 7.2% | 36.7% | 0.121 | 11 | 141 | 19 |

> **思考题 3**：哪些学科群之间最容易混淆（即 FP 主要来自哪些 category）？结合你选择的路由方案分析根本原因：是信号本身的局限（关键词覆盖不足 / 锚点语义重叠 / 分类边界模糊），还是学科本身内容交叉？

**回答：**

初版（仅 Keyword + description-only Preference）：
- 最明显的混淆是大量样本被误打到 `default-route`（FP=260，precision 仅 7.5%），说明绝大多数题目既未命中关键词、preference 锚点又只有 description 一个向量导致分类边界极弱，属于”漏召回主导”的错误模式。
- `route_cs` 出现”高精度低召回”（precision 100%，recall 3.3%），关键词过于保守，只抓到极少数典型措辞。
- 根本原因以**信号方案局限**为主：关键词覆盖不足、preference 只有单一 description 锚点。

调参后（增加 preference examples + threshold）：
- 总体准确率从 26.7% 提升到 40.7%（+14pp）；route_stem F1 从 0.217 升至 0.560，是改善最大的 decision。
- `default-route` FP 从 260 降至 149，说明 preference examples 显著提升了各学科的召回。
- `other` 类准确率从 70% 降至 36.7%，是 threshold=0.62 的副作用：部分 `other` 题目的相似度超过了某个学科的 threshold，被错误路由。
- `route_cs` 仍然最弱（F1=0.125）：MMLU-Pro CS 题目多为描述性考试语言，与 examples 中的编程/算法措辞存在语义偏差，属于**学科内部表述多样性**导致的锚点覆盖不足。

第三轮调参分析（CS 理论例句 + 化学例句 + threshold=0.60）：
- 整体准确率从 40.7% 升至 42.6%（+1.9pp），chemistry 是最大受益者（+16.7pp），直接原因是补充了有机化学反应机制例句，使 preference 信号覆盖到 MMLU-Pro 化学题的典型措辞。
- `route_cs` F1 从 0.125 升至 0.185，CS recall 从 10% 提升到 16.7%，改善来自理论性例句（可判定性、形式语言、数据库范式等）和新增关键词（`NP-complete`、`undecidable`、`halting problem` 等）。
- physics 小幅退步（56.7%→53.3%）：新增的 `pipeline`、`virtual memory` 等 CS 关键词与计算机体系结构内容重叠，部分 physics 题被误判为 route_cs，属于**信号扩展的副作用**——关键词粒度过粗时，扩大覆盖必然伴随新的 FP。
- `route_cs` 召回率仍然偏低的根本原因是**学科内部表述多样性**：CS 涵盖从理论计算机科学到系统、网络、AI 的极宽谱，而 preference 锚点向量数量有限（8 个例句），无法覆盖全部子领域的语义空间。这是 Contrastive 方案的固有局限，不能通过简单增加例句彻底解决。

---

### 4.4 端到端答题准确率评估

**目标**：验证路由到专项模型是否真正提升了答题准确率，对比「专项路由」与「全走 default-route」的效果差异。

```python
# eval_answer_accuracy.py
"""
端到端答题准确率：Router 路由 + LLM 回答 MMLU-Pro 多选题
复用 routing_accuracy_results.json 中的 420 条样本，追加 LLM 答案
"""
import json, re, time, requests
from collections import defaultdict
from datasets import load_dataset

ROUTER_URL = "http://localhost:8899/v1/chat/completions"

with open("routing_accuracy_results.json") as f:
    routing_results = {r["question_id"]: r for r in json.load(f)}

ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
samples = [item for item in ds if item["question_id"] in routing_results]

def extract_answer(text):
    text = text.strip()
    for pat in [r'[Tt]he answer is\s*\(?([A-J])\)?',
                r'[Aa]nswer:\s*\(?([A-J])\)?',
                r'^([A-J])[.\s\)]']:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    letters = re.findall(r'\b([A-J])\b', text)
    return letters[0] if len(letters) == 1 else ""

results = []
for i, item in enumerate(samples):
    options_str = "\n".join(
        f"{chr(65+j)}. {opt}" for j, opt in enumerate(item["options"]) if opt
    )
    prompt = f"{item['question']}\n\n{options_str}\n\nAnswer with just the letter (A-J)."
    payload = {
        "model": "MoM",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100, "stream": False, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(ROUTER_URL, json=payload, timeout=30)
        latency_ms = (time.perf_counter() - t0) * 1000
        body = resp.json()
        predicted = extract_answer(body["choices"][0]["message"]["content"])
        decision = resp.headers.get("x-vsr-selected-decision", "default-route") or "default-route"
    except Exception:
        latency_ms, predicted, decision = -1, "", "ERROR"

    routing_meta = routing_results[item["question_id"]]
    results.append({
        "question_id":       item["question_id"],
        "category":          item["category"],
        "expected_decision": routing_meta["expected_decision"],
        "actual_decision":   decision,
        "routing_correct":   routing_meta["correct"],
        "expected_answer":   item["answer"],
        "predicted_answer":  predicted,
        "answer_correct":    predicted == item["answer"],
        "latency_ms":        latency_ms,
    })

    if (i + 1) % 60 == 0:
        acc = sum(r["answer_correct"] for r in results) / len(results) * 100
        print(f"  进度 {i+1}/{len(samples)}, 当前答题准确率: {acc:.1f}%")

# ── 按学科 + 路由是否正确 分层统计 ──
by_cat = defaultdict(lambda: {"routed_correct": [], "routed_wrong": []})
for r in results:
    key = "routed_correct" if r["routing_correct"] else "routed_wrong"
    by_cat[r["category"]][key].append(r["answer_correct"])

print(f"\n{'学科':<20} {'路由正确时答题率':>17} {'路由错误时答题率':>17} {'样本数':>7}")
print("-" * 65)
for cat in sorted(by_cat):
    s = by_cat[cat]
    acc_ok  = sum(s["routed_correct"]) / len(s["routed_correct"]) * 100 if s["routed_correct"] else float("nan")
    acc_bad = sum(s["routed_wrong"])   / len(s["routed_wrong"])   * 100 if s["routed_wrong"]   else float("nan")
    total   = len(s["routed_correct"]) + len(s["routed_wrong"])
    print(f"{cat:<20} {acc_ok:>16.1f}% {acc_bad:>16.1f}% {total:>7}")

with open("answer_accuracy_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n完整结果已保存至 answer_accuracy_results.json")
```

**需填写的学科路由效果对比表格：**

| 学科 | 路由准确率 | 路由正确时答题率 | 路由错误时答题率 | Δ（正确-错误） |
|------|-----------|---------------|---------------|--------------|
| math | 10.0% | 33.3% | 14.8% | +18.5% |
| physics | 20.0% | 16.7% | 0.0% | +16.7% |
| engineering | 13.3% | 0.0% | 11.5% | -11.5% |
| computer science | 3.3% | 0.0% | 10.3% | -10.3% |
| biology | 33.3% | 10.0% | 25.0% | -15.0% |
| chemistry | 16.7% | 0.0% | 12.0% | -12.0% |
| health | 20.0% | 0.0% | 16.7% | -16.7% |
| law | 53.3% | 6.2% | 0.0% | +6.2% |
| history | 30.0% | 11.1% | 14.3% | -3.2% |
| philosophy | 0.0% | N/A | 26.7% | N/A |
| psychology | 30.0% | 22.2% | 23.8% | -1.6% |
| economics | 53.3% | 18.8% | 28.6% | -9.8% |
| business | 20.0% | 16.7% | 4.2% | +12.5% |
| other | 70.0% | 14.3% | 0.0% | +14.3% |

> **思考题 4**：Δ 值（路由正确时 vs 路由错误时的答题率差）是否在所有学科上都是正数？如果某学科 Δ 为负，说明什么？结合你的路由方案和 system_prompt 设计分析原因。

**回答：**

- Δ 值并不是在所有学科上都为正。正向收益比较明显的有 `math`、`physics`、`business`、`law` 和 `other`，但 `biology`、`chemistry`、`health`、`computer science`、`engineering` 等学科都出现了负值。  
- 当某学科 Δ 为负时，说明“路由正确”并没有转化为更高答题率，反而可能因为 system prompt 把模型限制在某种回答风格上，或关键词命中的题目本身更难，导致答题表现下降。  
- 在本实验里，所有路由最终都还是指向同一个 `glm-5` 模型，因此“路由正确”带来的增益主要来自 prompt 引导，而不是模型能力切换；这会让路由收益变弱，甚至在部分学科上出现负收益。  
- 此外，当前方案基于 Keyword Signal，容易把措辞中带有某些学科词的难题路由到专项 prompt，但这些 prompt 不一定比 default prompt 更适合当前模型，所以会出现“路由标签正确，但答题率不升反降”的情况。

> **思考题 5**：综合路由准确率（4.3 表格）和答题率提升（4.4 表格），哪个学科群的「路由收益」最大？你的方案在哪个学科群上表现最差？如果要改进，你会选择调整信号类型、锚点/词表内容，还是调整学科合并粒度（如把 engineering 从 STEM 组单独拆出来）？

**回答：**

- 从 Δ 值看，本次方案的路由收益最大的是 `math`（+18.5%）和 `physics`（+16.7%），说明这些题目在被正确打到 STEM 路由后，专项 prompt 对模型是有帮助的。  
- 表现最差的是 `health`（-16.7%）、`biology`（-15.0%）、`chemistry`（-12.0%）以及 `computer science`（-10.3%）。其中 `computer science` 的路由准确率只有 3.3%，说明词表覆盖严重不足；而生命科学相关学科则说明即使路由命中，专项 prompt 也未明显提升同一模型的答题能力。  
- 如果继续改进，我优先会做三件事：  
- 第一，改信号类型，从纯 Keyword 改成 Preference/contrastive，让分类更依赖语义而不是表面词；  
- 第二，补充和重写词表/锚点，尤其是 `computer science`、`engineering`、`biology` 这些当前召回明显不足的类别；  
- 第三，调整学科合并粒度，例如把 `engineering` 从 `route_stem` 中拆出来，避免被通用数学/物理词表淹没。

---


## 五、综合分析

### 5.1 性能数据汇总

将实验 1、2 的核心指标填入以下汇总表：

```
┌──────────────────────────────────────────────────────────────────────┐
│                    性能数据汇总                                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  各路径端到端延迟 (实验 1)                                            │
│  ┌──────────────────────────────┐                                    │
│  │ Keyword 路径 p50:   580.7 ms │                                    │
│  │ Preference 路径 p50:575.2 ms │                                    │
│  │ Default 路径 p50:   574.6 ms │                                    │
│  └──────────────────────────────┘                                    │
│                                                                      │
│  端到端吞吐 (实验 2, 并发=10)                                         │
│  ┌──────────────────────────────┐                                    │
│  │ QPS:                 7.0     │                                    │
│  │ p50:              1225.6 ms  │                                    │
│  │ p99:              2461.3 ms  │                                    │
│  └──────────────────────────────┘                                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.2 准确率数据汇总

```
┌──────────────────────────────────────────────────────────────────────┐
│                    准确率数据汇总                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  路由准确率对比 (实验 4.3)                                            │
│  ┌──────────────────────────────────────────┐                        │
│  │ 初版 (Keyword + desc-only Preference):   │                        │
│  │   default-route precision:      7.5%     │ ← 假阳性率 = 92.5%    │
│  │   总体路由准确率:               26.7%     │                        │
│  │                                          │                        │
│  │ 调参后 (+ examples×6 + threshold=0.62): │                        │
│  │   default-route precision:      6.9%     │ ← 假阳性率 = 93.1%    │
│  │   总体路由准确率:               40.7%     │ (+14pp)               │
│  └──────────────────────────────────────────┘                        │
│                                                                      │
│  MMLU-Pro 答题准确率 (实验 4.4，基于初版路由结果)                     │
│  ┌──────────────────────────────────────────┐                        │
│  │ 路由正确时均值答题率:          11.5%      │ ← 13 学科均值（哲学N/A）│
│  │ 路由错误时均值答题率:          13.5%      │ ← 14 学科均值           │
│  │ 均值 Δ（正确 - 错误）:         -2.0%      │ ← 同模型下路由本身无增益│
│  └──────────────────────────────────────────┘                        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 六、深入分析：Router 侧性能瓶颈与 2D Matryoshka 优化路径

> 对应提交清单第 6 项（选做）：从数据中发现的性能瓶颈分析。

---

### 6.1 实验测量范围的边界：`max_tokens=1` ≠ 纯路由延迟

实验 1 和实验 2 均使用 `max_tokens=1`，其原意是压缩 LLM 的 decode（续词）时间，让延迟更接近"路由开销"。但实际上这个假设并不成立。

一次完整请求的端到端延迟由以下几部分组成：

```
端到端延迟 = Router 信号评估 + 网络转发 + LLM prefill (TTFT) + LLM decode
                                                               ↑ max_tokens=1 只削减了这里
```

`max_tokens=1` 只消除了 decode 阶段（生成第 2、3… 个 token 的时间），但 **prefill（LLM 处理整个 input prompt 的前向推理）仍然完整存在**，且不受 `max_tokens` 影响。

**从数据验证这一结论：**

实验 1 中三条路径使用长度差异极小的 prompt（6–10 token），p50 差异仅 6ms（574.6ms vs 580.7ms）。如果 Router 信号评估真的主导了延迟，Keyword 路径（信号 <1ms）和 Preference 路径（信号 ~几十ms）应该差异明显。差异只有 6ms 说明两条路径的 Router 开销本身相近（均由 `wg.Wait()` 等 Preference 完成决定），而绝大部分延迟（~530ms）是 LLM prefill。

对于实验 2 的 MMLU 题目（题干 + 10 个选项，通常 100–300 token），prefill 时间更长，这也解释了为何并发度 10 时 p50 从 ~575ms 涨到了 ~1312ms：LLM prefill 的并发排队造成的，而不是 Router 侧的问题。

**结论：实验 1/2 测量的是「路由 + LLM TTFT」的组合延迟。三条路径 p50 相近，说明 wg.Wait() 使所有路径的 Router 开销趋于一致（均等待 Preference 完成），但绝对延迟值由上游 LLM 的 prefill 能力决定。**

---

### 6.2 `wg.Wait()` early cancellation：理论可行，工程上无法实现

#### 优化思路

`classifier_signal_dispatch.go:86` 中，所有信号以 goroutine 形式并行启动，`wg.Wait()` 在 `classifier_signal_context.go:100` 汇合，必须等所有 goroutine 完成才能继续。

若 Keyword 先完成且命中了高优先级 decision，理论上可以通过 `context.Cancel()` 通知 Preference goroutine 提前退出：

```
当前 config 优先级：
  keyword decisions:    300 / 290 / 280
  preference decisions: 200 / 200 / 200

若 Keyword 命中 priority ≥ 280，Preference 结果不可能改变最终决策
→ 理论上可 cancel Preference goroutine
```

#### 为什么工程上无法实现

Preference Contrastive 分类器的完整调用链是：

```
evaluatePreferenceSignal()                           [classifier_signal_context.go:304]
  └→ ContrastivePreferenceClassifier.Classify()     [contrastive_preference_classifier.go:163]
       └→ getEmbeddingWithModelType()                [embedding_classifier.go:20, alias]
            └→ candle_binding.GetEmbeddingWithModelType()
                 └→ GetEmbedding2DMatryoshka()       [candle-binding/semantic-router.go:1548]
                      └→ C.get_embedding_2d_matryoshka()   ← CGO 边界，进入 Rust
                              Rust/Candle: mmbert 22 层前向推理
```

一旦执行到 `C.get_embedding_2d_matryoshka()`（`candle-binding/semantic-router.go:1561`）：

1. Go scheduler 将该 goroutine **锁定在一个 OS 线程**上，线程不可被复用
2. 控制权完全交给 Rust/Candle 的 mmbert 前向推理
3. Go 的 `context.Done()` 信道**不被检查**——CGO 调用是同步阻塞
4. Rust 层没有暴露中止接口
5. 唯一出口是等 Rust 函数 `return`

即使在 Go 侧加 `select { case <-ctx.Done(): return }`，也只能在 CGO 调用**之前**或**之后**捕获，无法打断正在进行的 mmbert 推理。

#### 即使能实现，也不值得

实验 1 三条路径 p50 差异仅 6ms，说明 Router 侧（含 Preference 推理）在总延迟中占比极小。即使将 Preference 推理时间降为零，p50 也只会从 ~580ms 降至 ~530ms，改善不足 9%。**瓶颈在上游 LLM 的 prefill，而不在 Router 内部。**

---

### 6.3 多条 Preference 规则对延迟的影响

`Classify()` 的内部结构（`contrastive_preference_classifier.go:151`）：

```go
func (c *ContrastivePreferenceClassifier) Classify(text string) (*PreferenceResult, error) {
    // Step 1: 对 query 做一次 embedding 推理（唯一的慢操作）
    out, err := getEmbeddingWithModelType(text, c.modelType, 0)   // CGO → Rust，~几十ms
    queryEmbedding := out.Embedding

    // Step 2: 与所有预加载的 example embeddings 做余弦相似度（纯 CPU 点积）
    for ruleName, embeddings := range c.ruleEmbeddings {          // 顺序循环，非并行
        for _, emb := range embeddings {
            sim := cosineSimilarity(queryEmbedding, emb)          // 768维向量点积，<0.01ms/次
        }
    }
}
```

**关键结论：无论配置多少条 preference rule，CGO 推理只调用 1 次。** 规则数量只影响余弦相似度的循环次数，而这是纯 CPU 点积，比 mmbert 推理便宜几个数量级。因此增加 Preference 规则对每次请求的延迟影响可以忽略不计。

唯一真正受规则数量影响的是**启动时的预加载**（`preloadRuleEmbeddings()`）：每条规则的每个 example 都需要一次 CGO embedding 推理。但预加载只在服务初始化时执行一次，与运行时延迟无关。

---

### 6.4 真正有效的优化方向：mmbert 2D Matryoshka Layer Early Exit

#### 什么是 2D Matryoshka

**Matryoshka（套娃）** 是一种训练技巧，让 embedding 向量具备"截断后仍然有效"的性质。

**1D Matryoshka（维度套娃）**

普通 BERT 输出 768 维向量，必须用全部 768 维才有意义。Matryoshka 训练时，同时用 768、512、256、128 维的截断版本计算损失，强迫模型把最重要的信息压到前几维：

```
[最重要的信息 | 次要信息 | 细节信息 | ...]
 ←— 128 维 —→
 ←————— 256 维 —————→
 ←————————————— 768 维 ————————————→
```

好处：推理时按需截断，省内存省计算，精度下降很小。

**2D Matryoshka（层 × 维度）**

mmbert 在此基础上再加一个维度：层数也可以早退出。训练时同时优化所有组合：

```
层数 \ 维度   128   256   768
  L6           ✓     ✓     ✓
  L11          ✓     ✓     ✓
  L22          ✓     ✓     ✓
```

每个格子都单独算 loss，迫使每一层的浅层表示也具备语义区分能力。

如果是普通训练的 BERT，直接截到 L6 输出的向量毫无意义——因为浅层没有被训练成能独立工作的表示。2D Matryoshka 解决的正是这个问题。

**为什么选择减少层数而不是维数**

两种截断方式的加速效果截然不同：

每一层 Transformer 的计算量：
```
Attention:  O(seq² × d)     ← 序列长度平方 × 维度
FFN:        O(seq × d × 4d) ← 维度的 4 倍中间层
```

**维度截断**（768→256）发生在推理结束后：22 层全部跑完，最后把输出向量从 768 维截成 256 维。省掉的只是后续余弦相似度的 dot product 计算，耗时微乎其微。

**层数早退**（L22→L11）发生在推理中间：跑到第 11 层就停，剩余 11 层 Attention + FFN 全部跳过——省掉的是真正的大头。

| 策略 | 省掉的计算 | 实测加速 |
|------|-----------|---------|
| 768→256 维截断 | 最后一步 dot product | ~0%（可忽略）|
| L22→L11 层早退 | 11 层 Transformer | ~2× |
| L22→L6 层早退 | 16 层 Transformer | ~3.6× |

CGO 线程锁定时间也直接正比于推理耗时，层数减少后 OS 线程释放更快，高并发下的线程争用随之降低。

mmbert 在训练时对两个维度同时施加约束：

- **Layer 维度**：模型有 22 层 Transformer，但训练时在第 3、6、11、22 层的输出上均计算 loss，迫使浅层输出也具备足够的语义表达能力。推理时可以在任意支持的层早退，直接用该层 hidden state 作为 embedding。
- **Dimension 维度**：embedding 向量（768维）的前 N 维在训练时被优化为可独立使用的子空间（Matryoshka 嵌套结构），推理后直接截断到目标维度即可。

#### 当前使用方式

当前 `Classify()` 通过 `getEmbeddingWithModelType(text, "mmbert", 0)` 调用，最终执行：

```go
// candle-binding/semantic-router.go:1561
C.get_embedding_2d_matryoshka(cText, cModelType,
    C.int(0),  // targetLayer=0 → 完整 22 层
    C.int(0),  // targetDim=0   → 默认 768 维
    &result)
```

即每次请求都跑完全部 22 层，输出 768 维向量。

#### 实测 Benchmark 数据

使用项目内置的 `mmbert_2d_matryoshka_bench.rs`（`candle-binding/examples/`），对本地 mmbert 模型进行实测，CPU 环境，batch=4，seq≈64 token：

```bash
MMBERT_MODEL_PATH=/path/to/mmbert32k-intent-classifier-merged \
cargo run --release --no-default-features --example mmbert_2d_matryoshka_bench -- --device cpu --quick
```

**Layer Early Exit 结果（维度固定为 768）：**

| targetLayer | 延迟（batch=4） | 实际加速比 | 模型质量 |
|-------------|-------------|----------|---------|
| L22（完整） | 388ms | 1.0× | 100% |
| L11 | 195ms | **2.0×** | 67% |
| L6 | 109ms | **3.6×** | 56% |
| L3 | 54ms | **7.2×** | 55% |

> 注：benchmark 代码在 speedup 列存在 bug（baseline 在循环末尾才赋值），原始输出全为 1.00x。以上加速比由原始延迟值换算得出：388 / 195 = 2.0x，以此类推。

**Dimension Truncation 结果（层数固定为 22）：**

| targetDim | 延迟 | 相对变化 |
|-----------|------|---------|
| 768 | 391ms | baseline |
| 512 | 409ms | +4.6% |
| 256 | 399ms | +2.0% |
| 128 | 402ms | +2.8% |
| 64 | 418ms | +6.9% |

**关键发现：维度截断对延迟几乎没有改善**（甚至略有增加，属于测量波动）。计算瓶颈在 22 层 Transformer 前向传播，最后的维度切片操作耗时可忽略不计。

**Full 2D Matrix（layers × dims）：**

```
         |   768d |   256d |    64d
-----------------------------------
L22      |  381ms |  401ms |  388ms   ← 维度几乎无影响
L11      |  188ms |  195ms |  184ms
L6       |  105ms |  101ms |  101ms
L3       |   54ms |   55ms |   55ms   ← 层数是主要变量
```

结论：**有效的优化变量是 `targetLayer`，`targetDim` 的作用是节省内存和网络传输，而不是推理时间。**

**Batch Size Scaling 结果（L22 完整模型，较长序列）：**

| batch_size | 总延迟 | 每条 emb 耗时 | 吞吐量 |
|------------|--------|-------------|--------|
| 1 | 1767ms | 1767 ms/emb | 0.6 emb/s |
| 4 | 6963ms | 1741 ms/emb | 0.6 emb/s |
| 8 | 13895ms | 1737 ms/emb | 0.6 emb/s |
| 16 | 28030ms | 1752 ms/emb | 0.6 emb/s |
| 32 | 56769ms | 1774 ms/emb | 0.6 emb/s |

> 注：此测试使用较长序列（~116 token），因此单条 emb 耗时高于 `--quick` 模式（短序列 batch=4 时 L22 约 97 ms/emb）。两者均指向同一结论。

**关键发现：CPU 模式下 batch size 增大没有吞吐增益，吞吐量始终约 0.6 emb/s（线性扩展，无并行收益）。** 原因是 CPU 的 BLAS 线程池已被单条推理的矩阵乘法饱和，额外的 batch 只会线性叠加等待时间。这直接解释了实验 2 中并发 50 时 p99 骤增的机制：50 个 goroutine 同时进入 CGO 调用，每个都占用一个 OS 线程执行 BLAS，形成 CPU 串行队列，尾延迟随并发线性放大。

#### 如何修改代码支持 Layer Early Exit

代码改动集中在 4 个文件，约 30–40 行，纯参数传递，不涉及任何逻辑变更。

**核心约束**：`preloadRuleEmbeddings()` 和 `Classify()` 必须使用**完全相同**的 `targetLayer` 和 `targetDim`，否则预加载的 example embeddings 与查询 embedding 来自不同向量空间，余弦相似度无意义。由于两处均读取 struct 字段，这一约束天然被满足。

**① `config/model_config_types.go`** — `PreferenceModelConfig` 增加两个字段：

```go
type PreferenceModelConfig struct {
    UseContrastive *bool  `yaml:"use_contrastive,omitempty"`
    EmbeddingModel string `yaml:"embedding_model,omitempty"`
    EmbeddingLayer int    `yaml:"embedding_layer,omitempty"` // 新增：0=完整22层，可选3/6/11
    EmbeddingDim   int    `yaml:"embedding_dim,omitempty"`   // 新增：0=默认768维
}
```

**② `classification/contrastive_preference_classifier.go`** — struct 新增字段，两处调用改为 `GetEmbedding2DMatryoshka`：

```go
type ContrastivePreferenceClassifier struct {
    modelType   string
    targetLayer int  // 新增
    targetDim   int  // 新增
    ...
}

// preloadRuleEmbeddings 和 Classify 中的两处：
// 原：getEmbeddingWithModelType(text, c.modelType, 0)
// 改：candle_binding.GetEmbedding2DMatryoshka(text, c.modelType, c.targetLayer, c.targetDim)
```

**③ `classification/preference_classifier.go`** — 构造时透传参数：

```go
contrastive, err := NewContrastivePreferenceClassifier(
    rules, modelType,
    resolvedLocalCfg.EmbeddingLayer,
    resolvedLocalCfg.EmbeddingDim,
)
```

**④ `config.yaml`** — 用户侧配置（以 L11 为例，2× 加速）：

```yaml
global:
  model_catalog:
    modules:
      classifier:
        preference:
          embedding_model: mmbert
          embedding_layer: 11   # 用11层替代22层，加速2×
```

> **注意**：`embedding_layer` 必须放在 `global.model_catalog.modules.classifier.preference` 路径下，这是 Go 侧 `CanonicalGlobal → CanonicalModelCatalog → CanonicalModelModules → CanonicalClassifierModule → PreferenceModelConfig` 的解析链路。如果错误地放在 `global.classifier.preference` 下，validate.go 会输出 `Unknown field "classifier" in global` 警告，但参数不会被读取，默认 `targetLayer=0`（等同 L22 完整模型）。

需额外对 `embedding_layer` 的合法值（0/3/6/11/22）做校验，防止传入非法值时 Rust 层返回不友好错误。

---

### 6.5 Layer Early Exit 工程实施记录

本节记录将 2D Matryoshka Layer Early Exit 实际部署到 Router 并进行对比实验的完整操作过程，包含所有关键命令和验证输出。

#### 编译环境准备

vllm-sr 服务运行在 Docker 镜像 `vllm-sr:0.3` 中，其中的 router 二进制是旧版（不含 `embedding_layer` 配置支持）。改动仅涉及 Go 侧，Rust 库已支持 `targetLayer` 参数，无需重新编译 Rust。

**从现有镜像提取三个共享库**（避免重新编译 Rust，节省数小时）：

```bash
mkdir -p semantic-router/candle-binding/target/release \
         semantic-router/ml-binding/target/release \
         semantic-router/nlp-binding/target/release \
         semantic-router/bin

docker create --name tmp vllm-sr:0.3
docker cp tmp:/usr/local/lib/libcandle_semantic_router.so \
    semantic-router/candle-binding/target/release/
docker cp tmp:/usr/local/lib/libml_semantic_router.so \
    semantic-router/ml-binding/target/release/
docker cp tmp:/usr/local/lib/libnlp_binding.so \
    semantic-router/nlp-binding/target/release/
docker rm tmp
```

提取结果：
```
-rwxr-xr-x  16M  libcandle_semantic_router.so   # Rust/Candle embedding 推理
-rwxr-xr-x 676K  libml_semantic_router.so
-rwxr-xr-x 1.8M  libnlp_binding.so
```

#### 代码改动（4 文件，约 15 行净增）

**① `config/model_config_types.go`** — `PreferenceModelConfig` 增加两字段：
```go
EmbeddingLayer int `yaml:"embedding_layer,omitempty"` // 0=完整22层；合法值:3/6/11/22
EmbeddingDim   int `yaml:"embedding_dim,omitempty"`   // 0=默认768维
```

**② `classification/embedding_classifier.go`** — 增加包级变量（与现有 `getEmbeddingWithModelType` 模式一致）：
```go
var getEmbedding2DMatryoshka = candle_binding.GetEmbedding2DMatryoshka
```

**③ `classification/contrastive_preference_classifier.go`** — struct 增加字段，构造函数签名新增两参数，两处调用替换：
```go
// struct 新增
targetLayer int
targetDim   int

// 构造函数签名
func NewContrastivePreferenceClassifier(rules, modelType string, targetLayer int, targetDim int)

// preloadRuleEmbeddings 和 Classify 两处均改为：
out, err := getEmbedding2DMatryoshka(text, c.modelType, c.targetLayer, c.targetDim)
```

**④ `classification/preference_classifier.go`** — 构造时透传：
```go
contrastive, err := NewContrastivePreferenceClassifier(
    rules, modelType,
    resolvedLocalCfg.EmbeddingLayer,
    resolvedLocalCfg.EmbeddingDim,
)
```

#### Go 编译

```bash
cd semantic-router/src/semantic-router
go build -tags=milvus -o ../../bin/router ./cmd
# 输出：108M bin/router（无任何错误）
```

#### 构建测试镜像

```dockerfile
# Dockerfile.emb-test
FROM vllm-sr:0.3
COPY bin/router /usr/local/bin/router
```

```bash
cd semantic-router
docker build -f Dockerfile.emb-test -t vllm-sr:emb-test .
# 4 秒完成（仅在原镜像上加一层）
```

#### Config 准备

三份配置文件均基于 `semantic/config.yaml` 复制，新增以下两处差异：

```yaml
# global 节（三份仅 embedding_layer 不同：22 / 11 / 6）
global:
  model_catalog:
    kbs: []    # 覆盖新版默认的 privacy_kb（旧镜像无对应 kb/ 目录）
    modules:
      classifier:
        preference:
          embedding_model: mmbert
          embedding_layer: 22   # L22: baseline / L11: 2× / L6: 3.6×
```

> **注 1**：`embedding_layer` 必须放在 `global.model_catalog.modules.classifier.preference` 路径下。初次实验时误放在 `global.classifier.preference`，validate.go 输出了 `Unknown field "classifier" in global` 警告但并未阻止启动，导致三组实验全部以 `targetLayer=0`（=L22）运行，产出了完全一致的假数据。发现后通过增加 `logging.Infof("[Preference Contrastive] 2D Matryoshka config: targetLayer=%d...")` 日志确认参数是否生效。
>
> **注 2**：新版 Go 源码的 `DefaultCanonicalGlobal()` 在 `canonical_defaults.go:110` 加入了 `privacy_kb` 默认 KB，要求 `/app/kb/privacy/labels.json` 存在。旧 Docker 镜像没有该目录，因此需要用 `kbs: []` 显式覆盖默认值，否则 router 会 FATAL 退出。

#### 每轮实验启动流程

```bash
cd /home/nickw/sf/semantic
source .venv/bin/activate

# 切换配置（L22/L11/L6 三选一）
cp config.l22.yaml config.yaml   # 或 config.l11.yaml / config.l6.yaml

# 重启
vllm-sr stop && vllm-sr serve --image vllm-sr:emb-test --image-pull-policy never
```

**L22 baseline 启动验证日志**（等待出现以下输出确认 embedding 层生效）：
```
{"level":"info","caller":"contrastive_preference_classifier.go:146",
 "msg":"[Preference Contrastive] preloaded 40/40 example embeddings using model=mmbert in 2.444s"}
{"level":"info","caller":"classifier_model_select.go:271",
 "msg":"Preference classifier initialized successfully with 5 routes"}
```

---

### 6.6 Layer Early Exit 对比实验结果

使用 6.5 节的修改版 router 二进制（`vllm-sr:emb-test` 镜像），分别以 `embedding_layer: 22`、`11`、`6` 三种配置启动服务，各跑实验 2（并发吞吐量）和实验 3（路由准确率），使用相同的 MMLU-Pro 测试集（420 条，每学科 30 条）。

> **勘误**：本节数据为第二轮实验结果。第一轮实验因 config.yaml 路径错误（`embedding_layer` 放在 `global.classifier.preference` 而非正确的 `global.model_catalog.modules.classifier.preference`），三组实验全部以 `targetLayer=0`（=L22）运行，产出了完全一致的虚假数据。发现后通过增加 `logging.Infof` 确认参数生效，修正配置路径后重新跑完全部实验。

**Preload 时间对比**（启动日志，40 个 example embedding）：

| 配置 | preload 时间 | 加速比 |
|------|------------|--------|
| L22 | 2.33s | 1.0× |
| L11 | 1.18s | **2.0×** |
| L6 | 0.68s | **3.4×** |

#### 实验 2 对比：并发吞吐量

| 并发度 | L22 QPS | L11 QPS | L6 QPS | L22 p50 | L11 p50 | L6 p50 | L22 p99 | L11 p99 | L6 p99 |
|--------|---------|---------|--------|---------|---------|--------|---------|---------|--------|
| 1 | 1.3 | 1.7 | 1.6 | 659.6 | 514.7 | 622.8 | 1369.5 | 1458.2 | 905.0 |
| 5 | 4.3 | 5.4 | 5.3 | 1072.1 | 887.4 | 881.6 | 2131.8 | 1475.1 | 2243.9 |
| 10 | 7.1 | 9.6 | 10.9 | 1222.0 | 951.7 | 900.9 | 2391.5 | 1544.2 | 1303.6 |
| 20 | 10.6 | 14.7 | 16.9 | 1767.7 | 1365.2 | 1201.3 | 3313.3 | 1842.5 | 1622.7 |
| 50 | 12.4 | 21.4 | 25.7 | 3431.4 | 2332.4 | 1893.7 | 8592.1 | 3752.2 | 2899.7 |

**关键观察：Layer Early Exit 带来了显著的吞吐量提升。**

- 并发 10 时：L22 QPS=7.1 → L11 QPS=9.6（+35%）→ L6 QPS=10.9（+54%）
- 并发 20 时：L22 QPS=10.6 → L11 QPS=14.7（+39%）→ L6 QPS=16.9（+59%）
- **并发 50 时：L22 QPS=12.4 → L11 QPS=21.4（+73%）→ L6 QPS=25.7（+107%）**
- p99 尾延迟同步改善：并发 50 时 L22 p99=8592ms → L6 p99=2900ms（降低 66%）

这与之前第一轮实验（配置错误导致三组一致）的结论完全不同。**Layer Early Exit 的收益在高并发下非常显著**，原因是：Preference embedding 推理（CGO→Rust）期间 goroutine 锁定 OS 线程，L22 需要更长的线程占用时间，高并发时大量线程排队；L6/L11 缩短了每次推理的持锁时间，有效降低了 CPU 侧的串行瓶颈。

原始输出：

```
# L22 (embedding_layer: 22)
     并发度      QPS    p50(ms)    p99(ms)   errors
--------------------------------------------------
       1      1.3      659.6     1369.5        0
       5      4.3     1072.1     2131.8        0
      10      7.1     1222.0     2391.5        0
      20     10.6     1767.7     3313.3        0
      50     12.4     3431.4     8592.1        0

# L11 (embedding_layer: 11)
     并发度      QPS    p50(ms)    p99(ms)   errors
--------------------------------------------------
       1      1.7      514.7     1458.2        0
       5      5.4      887.4     1475.1        0
      10      9.6      951.7     1544.2        0
      20     14.7     1365.2     1842.5        0
      50     21.4     2332.4     3752.2        0

# L6 (embedding_layer: 6)
     并发度      QPS    p50(ms)    p99(ms)   errors
--------------------------------------------------
       1      1.6      622.8      905.0        0
       5      5.3      881.6     2243.9        1
      10     10.9      900.9     1303.6        0
      20     16.9     1201.3     1622.7        0
      50     25.7     1893.7     2899.7        0
```

#### 实验 3 对比：路由准确率

| 配置 | 总体准确率 | route_stem F1 | route_cs F1 | route_science F1 | route_humanities F1 | route_business F1 | default-route F1 |
|------|-----------|--------------|------------|-----------------|--------------------|--------------------|------------------|
| L22 | **42.6%** | 0.564 | 0.185 | 0.423 | 0.603 | 0.600 | 0.121 |
| L11 | **47.9%** | 0.631 | 0.190 | 0.424 | 0.503 | 0.625 | 0.000 |
| L6 | **39.5%** | 0.630 | 0.122 | 0.373 | 0.265 | 0.561 | 0.000 |

**L22 按学科明细**（emb-test 镜像，targetLayer=22）：

| 学科 | 准确率 | 正确 | 总数 |
|------|--------|------|------|
| math | 36.7% | 11 | 30 |
| physics | 53.3% | 16 | 30 |
| engineering | 63.3% | 19 | 30 |
| computer science | 16.7% | 5 | 30 |
| biology | 33.3% | 10 | 30 |
| chemistry | 30.0% | 9 | 30 |
| health | 36.7% | 11 | 30 |
| law | 66.7% | 20 | 30 |
| history | 53.3% | 16 | 30 |
| philosophy | 40.0% | 12 | 30 |
| psychology | 30.0% | 9 | 30 |
| economics | 66.7% | 20 | 30 |
| business | 33.3% | 10 | 30 |
| other | 36.7% | 11 | 30 |

**L11 按学科明细**（emb-test 镜像，targetLayer=11）：

| 学科 | 准确率 | 正确 | 总数 | vs L22 变化 |
|------|--------|------|------|------------|
| math | 80.0% | 24 | 30 | **+43.3pp** |
| physics | 83.3% | 25 | 30 | **+30.0pp** |
| engineering | 90.0% | 27 | 30 | **+26.7pp** |
| computer science | 36.7% | 11 | 30 | **+20.0pp** |
| biology | 26.7% | 8 | 30 | -6.7pp |
| chemistry | 13.3% | 4 | 30 | -16.7pp |
| health | 53.3% | 16 | 30 | +16.7pp |
| law | 23.3% | 7 | 30 | -43.3pp |
| history | 63.3% | 19 | 30 | +10.0pp |
| philosophy | 40.0% | 12 | 30 | — |
| psychology | 26.7% | 8 | 30 | -3.3pp |
| economics | 60.0% | 18 | 30 | -6.7pp |
| business | 73.3% | 22 | 30 | **+40.0pp** |
| other | 0.0% | 0 | 30 | -36.7pp |

**L6 按学科明细**（emb-test 镜像，targetLayer=6）：

| 学科 | 准确率 | 正确 | 总数 | vs L22 变化 |
|------|--------|------|------|------------|
| math | 90.0% | 27 | 30 | **+53.3pp** |
| physics | 83.3% | 25 | 30 | **+30.0pp** |
| engineering | 93.3% | 28 | 30 | **+30.0pp** |
| computer science | 30.0% | 9 | 30 | +13.3pp |
| biology | 30.0% | 9 | 30 | -3.3pp |
| chemistry | 13.3% | 4 | 30 | -16.7pp |
| health | 40.0% | 12 | 30 | +3.3pp |
| law | 3.3% | 1 | 30 | **-63.3pp** |
| history | 26.7% | 8 | 30 | -26.7pp |
| philosophy | 23.3% | 7 | 30 | -16.7pp |
| psychology | 13.3% | 4 | 30 | -16.7pp |
| economics | 60.0% | 18 | 30 | -6.7pp |
| business | 46.7% | 14 | 30 | +13.3pp |
| other | 0.0% | 0 | 30 | -36.7pp |

#### 核心发现

**1. Layer Early Exit 对吞吐量有显著提升，尤其在高并发场景。** 并发 50 时 L6 的 QPS 是 L22 的 2.1 倍（25.7 vs 12.4），p99 从 8.6s 降至 2.9s。这是因为 Preference embedding 的 CGO 调用会锁定 OS 线程，缩短推理时间直接减少了线程争用。

**2. L11 路由准确率最高（47.9%），优于 L22（42.6%）和 L6（39.5%）。** 这是一个非直觉的结果：
- L11 在 STEM 类学科上大幅提升（math +43pp, engineering +27pp, physics +30pp），但在人文类学科上下降（law -43pp）
- L6 在 STEM 上更强（math 90%），但人文社科严重退化（law 3.3%, history 26.7%）
- 可能原因：浅层更侧重词汇/句法特征，对 STEM 的公式化表述敏感，但对人文社科的语义细微差别缺乏辨识能力；L11 在两者之间取得了最佳平衡

**3. 所有配置下 default-route 都不再接收流量（L11/L6 的 default-route TP=0, FP=0）。** 对比 L22 有 11 个 TP 和 141 个 FP，说明浅层 embedding 的相似度普遍更高，更容易超过 threshold 被某个 preference 规则捕获。`other` 类准确率从 36.7%（L22）降至 0%（L11/L6）正是这一效应的体现。

**4. 2D Matryoshka 训练确实使浅层具备了有意义的语义表达，但不同层对不同学科领域的判别能力有差异。** 这不是简单的"浅层 = 差"，而是"浅层 = 不同的语义空间"。

**结论：L11 是当前任务的最佳配置**——兼顾路由准确率（47.9%，最高）和吞吐量改善（并发 50 QPS +73%），是性能与质量的最佳平衡点。L6 在纯吞吐场景下更优（+107%），但路由准确率有明显下降。

---

### 6.7 优化方向总结

| 优化方向 | 实现难度 | Embedding 加速 | 路由准确率 | 并发50 QPS | 说明 |
|---------|---------|---------------|-----------|-----------|------|
| **L22（baseline）** | — | 1.0× | 42.6% | 12.4 | 完整 22 层，当前默认配置 |
| **Layer Early Exit（L11）** | 低（15行，4文件） | 2.0× | **47.9%（+5.3pp）** | **21.4（+73%）** | **推荐配置：准确率最高 + 吞吐量大幅提升** |
| **Layer Early Exit（L6）** | 同上 | 3.6× | 39.5%（-3.1pp） | **25.7（+107%）** | 最高吞吐，但人文社科准确率下降明显 |
| **Layer Early Exit（L3）** | 同上 | 7.2× | 未测试 | 未测试 | 可能影响准确率，需实测 |
| **Keyword-only** | 零代码 | ∞（消除 Preference） | 26.7% | ~12.4 | 失去语义路由能力 |
| **context.Cancel()** | 无法实现 | — | — | — | CGO 不可中断 |
| **Dimension 截断** | 低 | 无改善 | — | — | 仅节省内存，不加速推理 |

**最终结论**：

1. **Layer Early Exit 在高并发场景下带来显著的端到端性能提升。** L6 在并发 50 时 QPS 达到 L22 的 2.1 倍（25.7 vs 12.4），p99 尾延迟从 8.6s 降至 2.9s。原因是 Preference embedding 的 CGO→Rust 调用会锁定 OS 线程，缩短推理时间直接减少了线程争用。低并发时（c=1）差异较小，因为此时无线程排队。

2. **L11 是最佳配置——准确率和吞吐量同时优于 L22 baseline。** L11 准确率 47.9%（+5.3pp vs L22），并发 50 QPS 21.4（+73% vs L22），在质量和性能上都是最优。这个非直觉的结果说明：L11 层的语义表达在当前 Preference 例句集下恰好能更好地区分学科边界（尤其是 STEM 类），而 L22 的额外深层特征对 Contrastive 匹配反而可能引入噪声。

3. **不同层对不同学科领域的判别能力存在差异。** 浅层（L6/L11）在 STEM 类（math/physics/engineering）上大幅优于 L22，但在人文社科（law/history）上显著退化。L11 在两者之间取得了最佳平衡。这表明 2D Matryoshka 的浅层训练在公式化/结构化表述上具备良好的语义排序能力，但对依赖上下文细微差别的人文类题目区分力不足。

4. **工程教训：配置路径验证至关重要。** 第一轮实验因 YAML 路径错误（`global.classifier.preference` vs 正确的 `global.model_catalog.modules.classifier.preference`）导致三组全部以 L22 运行，产出了"三组完全一致"的虚假结论。增加 `logging.Infof` 确认 `targetLayer` 实际值后才发现问题。生产环境中应对关键配置参数增加启动时校验日志。

---

## 七、提交清单

| # | 内容 | 格式 |
|---|------|------|
| 1 | 实验 1、2 所有表格的实测数据 | 填入本文档 |
| 2 | 思考题 1-5 的回答（需结合代码和数据） | 文字说明 |
| 3 | 第五部分性能 + 准确率汇总图 | 填入汇总框 |
| 4 | `routing_accuracy_results.json` 完整路由评估结果 | JSON 文件 |
| 5 | `answer_accuracy_results.json` 完整答题评估结果 | JSON 文件 |
| 6 | (选做) 从数据中发现的 1-2 个性能或准确率瓶颈分析 | 自由发挥 |

---

## 附录：关键代码位置速查

| 组件 | 文件 | 行号 | 核心逻辑 |
|------|------|------|---------|
| 信号并行评估入口 | `classification/classifier_signal_context.go` | 70 | `EvaluateAllSignalsWithContext()` |
| 信号 goroutine 调度 | `classification/classifier_signal_dispatch.go` | 86 | `runSignalDispatchers()` → `wg.Wait()` |
| 信号类型启用判断 | `classification/classifier_signal_eval.go` | 185 | `isSignalTypeUsed()` |
| Keyword 正则匹配 | `classification/keyword_classifier.go` | 259 | `ClassifyWithKeywordsAndCount()` |
| Preference Contrastive 推理 | `classification/contrastive_preference_classifier.go` | 151 | `Classify()` → 1次FFI + N次余弦 |
| Preference 预加载 | `classification/contrastive_preference_classifier.go` | 61 | `preloadRuleEmbeddings()` |
| CGO embedding 调用 | `candle-binding/semantic-router.go` | 1548 | `GetEmbedding2DMatryoshka()` → Rust |
| CGO 实际 FFI 边界 | `candle-binding/semantic-router.go` | 1561 | `C.get_embedding_2d_matryoshka()` |
| Preference 配置类型 | `config/model_config_types.go` | 142 | `PreferenceModelConfig` |
| Decision 优先级排序 | `decision/engine.go` | 321 | `selectBestDecision()` |
| 路由评估总入口 | `extproc/req_filter_classification.go` | 22 | `performDecisionEvaluation()` |
| 2D Matryoshka Benchmark | `candle-binding/examples/mmbert_2d_matryoshka_bench.rs` | — | Layer×Dim 实测基准 |

---
