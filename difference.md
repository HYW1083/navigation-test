# StreamVLN vs JanusVLN 性能差异分析

## 关键发现

**StreamVLN运行更快的主要原因不是场景分配策略，而是推理策略！**

实际上，StreamVLN使用的是**更慢**的场景切换策略，但通过**大幅减少模型推理次数**来获得整体性能提升。

---

## 详细对比

### 1. 场景分配策略 ❌ 不是主要差异

#### StreamVLN (streamvln_eval.py:215-228)
```python
# 使用旧的策略 - 遍历所有场景，但只处理部分episode
total_episodes = sum(len(scene_episode_dict[scene][idx::self.env_num])
                     for scene in sorted(scene_episode_dict.keys()))

for scene in sorted(scene_episode_dict.keys()):  # 遍历所有场景
    episodes = scene_episode_dict[scene]
    scene_episodes = episodes[idx::self.env_num]  # 每个进程处理部分episode
    for episode in scene_episodes:
        ...
```

**问题**: 每个进程会加载所有61个场景，但只处理其中的部分episodes，仍然会有大量场景切换

#### JanusVLN (evaluation_fast.py:168-191)
```python
# 使用新的优化策略 - 每个进程只处理分配的场景
all_scenes = sorted(scene_episode_dict.keys())
my_scenes = all_scenes[idx::self.env_num]  # 只分配部分场景

for scene_idx, scene in enumerate(my_scenes):  # 只遍历分配的场景
    episodes = scene_episode_dict[scene]
    for episode in episodes:  # 处理该场景的所有episodes
        ...
```

**优势**: 每个进程只加载~8个场景，场景切换次数减少22倍

**结论**: JanusVLN在场景切换方面更优，所以场景切换不是StreamVLN更快的原因。

---

### 2. 模型推理策略 ✅ 核心差异

#### StreamVLN: 批量动作生成 + KV缓存

**关键代码 (streamvln_eval.py:300-360)**:
```python
action_seq = []  # 动作队列
past_key_values = None  # KV缓存
output_ids = None

while not env.episode_over:
    # 只在动作队列为空时调用模型
    if len(action_seq) == 0:
        if output_ids is None:
            # 首次推理
            input_ids, conversations = self.preprocess_qwen([sources], ...)
        else:
            # 后续推理使用空prompt
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            input_ids = torch.cat([output_ids, input_ids.to(output_ids.device)], dim=1)

        # 调用模型生成多步动作
        outputs = self.model.generate(
            **input_dict,
            max_new_tokens=10000,  # 生成很多tokens
            use_cache=True,  # 启用KV缓存
            past_key_values=past_key_values  # 复用之前的KV
        )

        output_ids = outputs.sequences
        past_key_values = outputs.past_key_values  # 保存KV缓存

        # 解析生成的多个动作
        action_seq = self.parse_actions(llm_outputs)  # 例如: [1, 1, 2, 1, 3, 1, 1, 0]

    # 从队列中取出一个动作执行
    action = action_seq.pop(0)
    observations = env.step(action)
    step_id += 1

    # 每32步重置一次
    if step_id % self.num_frames == 0:
        self.model.reset_for_env(idx)
        output_ids = None
        past_key_values = None
```

**推理次数**:
- 假设`num_frames=32`, episode长度为400步
- 模型调用次数: 400 / 32 ≈ **12-13次**
- 每次生成32步的动作序列

#### JanusVLN: 每步推理

**关键代码 (evaluation_fast.py:248-270)**:
```python
while not env.episode_over:
    # 准备历史图像
    history_len = len(rgb_list) - 1
    if history_len <= self.num_history:
        history_images = rgb_list[:history_len]
        images = history_images + [rgb_list[-1]]
    else:
        indices = np.linspace(0, history_len, self.num_history + 1, dtype=int)
        images = [rgb_list[i] for i in indices]

    # 每一步都调用模型
    action = self.model.call_model(images, episode_instruction, step_id)[0]

    if action in self.actions2idx:
        action = self.actions2idx[action][0]
    else:
        action = 0

    observations = env.step(action)
    step_id += 1
```

**推理次数**:
- Episode长度为400步
- 模型调用次数: **400次**
- 每次只生成1个动作

**速度对比**:
```
StreamVLN: 12-13次模型调用 / episode
JanusVLN:  400次模型调用 / episode
速度差异:  ~30倍模型调用差异
```

---

### 3. 其他次要差异

#### 指标计算频率

**StreamVLN (streamvln_eval.py:295-298)**:
```python
# 每一步都计算指标
info = env.get_metrics()
if info['top_down_map'] is not None:
    frame = observations_to_image({'rgb':observations['rgb']}, info)
    vis_frames.append(frame)
```

**JanusVLN (evaluation_fast.py:242-246)**:
```python
# 只在保存视频时计算（默认5%的episodes）
if should_save_video:
    info = env.get_metrics()
    if info['top_down_map'] is not None:
        frame = observations_to_image({'rgb':observations['rgb']}, info)
        vis_frames.append(frame)
```

**结论**: JanusVLN在这方面更优（少调用95%的metrics计算）

#### 视频保存比例

**StreamVLN**: 保存所有episodes的视频（如果`--save_video`）
**JanusVLN**: 只保存5%的episodes（`--save_video_ratio 0.05`）

---

## 性能瓶颈分析

### JanusVLN当前瓶颈

根据上述分析，JanusVLN的主要性能瓶颈是：

1. **模型推理频率过高** (400次/episode)
   - 每步都调用`model.call_model()`
   - 每次调用都重新处理历史图像
   - 没有KV缓存，每次都从头计算attention

2. **历史图像处理开销**
   - 每步都重新采样和处理历史图像 (最多9张)
   - 图像预处理、特征提取都重复进行

3. **场景切换优化已做，但不是主要瓶颈**
   - 场景加载时间: ~80秒/进程
   - 模型推理时间: ~3600秒/进程 (假设10秒/episode × 168 episodes)
   - 模型推理占总时间的 **98%**

---

## 性能预估

### 假设条件
- 8个GPU进程
- 每个进程处理168个episodes
- 平均episode长度: 20步

### StreamVLN
```
模型调用次数: 168 episodes × (20 steps / 32 frames) = 105次
假设每次调用0.5秒: 105 × 0.5 = 52.5秒
场景切换时间: ~5分钟 (切换多)
总时间: ~7分钟
```

### JanusVLN (当前)
```
模型调用次数: 168 episodes × 20 steps = 3,360次
假设每次调用0.3秒: 3,360 × 0.3 = 1,008秒 = 16.8分钟
场景切换时间: ~1.3分钟 (优化后)
总时间: ~18分钟
```

### 预期差异
StreamVLN快 **2.5-3倍**，主要来自:
- 减少模型调用32倍
- 使用KV缓存加速推理

---

## 建议的优化方案

### 方案1: 实现批量动作生成（推荐）

**修改evaluation_fast.py**添加动作队列:

```python
action_queue = []

while not env.episode_over:
    if len(action_queue) == 0:
        # 只在队列为空时调用模型
        images = prepare_images(...)
        actions_text = self.model.call_model(images, episode_instruction, step_id)[0]

        # 解析多个动作: "MOVE_FORWARD, MOVE_FORWARD, TURN_LEFT, ..."
        action_queue = parse_multiple_actions(actions_text)

        # 限制最多32步
        action_queue = action_queue[:32]

    action = action_queue.pop(0)
    observations = env.step(action)
    step_id += 1
```

**预期效果**:
- 模型调用从400次降至12-15次/episode
- **加速25-30倍推理时间**
- 总评估时间从1-2小时降至5-10分钟

### 方案2: 添加KV缓存（需要模型支持）

修改模型生成方法，保留和复用past_key_values:

```python
past_key_values = None

while not env.episode_over:
    outputs = model.generate(
        ...,
        use_cache=True,
        past_key_values=past_key_values
    )
    past_key_values = outputs.past_key_values
```

**预期效果**:
- 减少重复计算历史帧的attention
- **加速2-3倍推理时间**

### 方案3: 降低推理频率

每N步调用一次模型，中间重复上一个动作:

```python
action = None
while not env.episode_over:
    if step_id % 4 == 0:  # 每4步推理一次
        action = self.model.call_model(...)

    observations = env.step(action)
```

**预期效果**:
- 模型调用减少75%
- **加速3-4倍**
- 可能轻微降低导航精度

---

## 总结

| 方面 | StreamVLN | JanusVLN | 优势方 |
|------|-----------|----------|--------|
| 场景分配策略 | 旧策略（切换多） | 优化策略（切换少） | JanusVLN ✅ |
| 模型推理频率 | 12-15次/episode | 400次/episode | StreamVLN ✅✅✅ |
| KV缓存 | 使用 | 不使用 | StreamVLN ✅ |
| 批量动作生成 | 使用 | 不使用 | StreamVLN ✅✅ |
| 指标计算 | 每步都算 | 按需计算 | JanusVLN ✅ |
| 整体速度 | 快 | 慢 | StreamVLN ✅✅✅ |

**关键发现**:
- StreamVLN快的核心原因是**批量动作生成 + KV缓存**，不是场景分配策略
- JanusVLN已经在场景分配上做了正确的优化，但模型推理策略仍是瓶颈
- 建议JanusVLN实现**方案1（批量动作生成）**可获得25-30倍加速
