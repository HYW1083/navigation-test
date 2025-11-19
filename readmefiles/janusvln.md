● 总结：JanusVLN核心设计实现

  基于对论文和代码的分析，JanusVLN的最核心设计写在 /data/home/co/cohw2/
  Projects/JanusVLN/src/qwen_vl/model/modeling_qwen2_5_vl.py
  文件中，特别是 Qwen2_5_VLForConditionalGenerationForJanusVLN
  类（第1651行开始）。

  核心设计组件及位置：

  1. 双隐式记忆架构 (Dual Implicit Memory)

  - 空间-几何记忆：
    - VGGT集成（第1659-1665行）：作为3D空间几何编码器
    - VGGT的KV缓存（第1663行）：self.past_key_values_vggt
    - 在forward中的使用（第2014-2028行）
  - 视觉-语义记忆：
    - Qwen2.5-VL视觉编码器（第1658行）：self.visual
    - 标准LLM的KV缓存（第2116-2127行传递的past_key_values）

  2. 混合增量更新策略 (StartRecentKVCache，第1607-1648行)

  self.kv_cache_vggt = StartRecentKVCache(
      start_size=8,      # 保留初始8帧
      recent_size=48,    # 滑动窗口48帧  
      k_seq_dim=2,
      v_seq_dim=2
  )
  这实现了论文中的"initial and sliding window"策略。

  3. 特征融合机制 (第2048行)

  image_embeds = image_embeds + self.lam * image_embeds_3d
  这对应论文中的方程4：Ft = S't + λ * MLP(G't)
  - image_embeds：语义特征
  - image_embeds_3d：经过VGGTMerger处理的空间特征
  - self.lam = 0.2：空间特征权重λ

  4. VGGTMerger投影层 (第149-164行)

  将VGGT的空间特征投影到与语义特征相同的维度，实现特征融合。

  关键创新点：

  1. 解耦设计：将语义理解和空间感知分离为两个独立通道
  2. 固定大小记忆：避免记忆随轨迹长度增长
  3. 避免重复计算：通过KV缓存机制，无需重新处理历史帧
  4. 高效推理：相比原始VGGT，推理时间降低69%-90%（论文Table 5）

  这种双隐式记忆架构是JanusVLN的核心创新，使其在VLN任务上达到了SOTA性能
