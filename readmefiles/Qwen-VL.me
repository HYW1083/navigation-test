# Qwen-VL 数据处理全流程解析

本文档详细解释了从**原始图片/视频**到最终的 **`input_ids`** 和 **`image_tokens`** 的转换过程。

## 1. 整体流程概览

整个过程可以分为两个并行的分支：文本分支（生成 `input_ids`）和视觉分支（生成 `image_tokens`）。

```mermaid
graph TD
    A[原始输入 (Text + Image)] --> B{Processor (Qwen3VLProcessor)}
    B -->|Text Pipeline| C[Tokenizer]
    B -->|Vision Pipeline| D[Image Processor]
    
    C --> E(Input IDs 序列)
    D --> F(Pixel Values)
    
    F --> G[Vision Encoder (Qwen3VLVisionModel)]
    G --> H(Image Tokens / Embeddings)
    
    E --> I[Model Forward]
    H --> I
    I --> J[Multimodal Fusion]
```

---

## 2. 详细变换步骤

### A. 文本/骨架生成 (`input_ids` 的由来)

1.  **原始输入**:
    *   用户输入: `"Describe this image: <img>path/to/img.jpg</img>"`
    *   系统将其格式化为通过特殊标记包裹的序列。

2.  **特殊 Token 插入**:
    *   Processor 会检测输入中的图片，并将其替换为一系列特殊的 **占位符 Token**。
    *   格式: `<|vision_start|>` + `N 个 <|image_pad|>` + `<|vision_end|>`
    *   **N 的计算**: 取决于图片分辨率。例如，Qwen-VL 通常将图片切分为 $14 \times 14$ 的 patch，加上重叠和池化，最终可能生成如 256 个 token。

3.  **Tokenization (分词)**:
    *   文本部分被切分为普通的 Token ID (如 `Move`: 151644)。
    *   图片占位符被转换为 `151655` (`<|image_pad|>`)。
    *   最终产出 **`input_ids`**: `[151644, 8948, ..., 151652, 151655, 151655, ..., 151653]`

### B. 视觉特征生成 (`image_tokens` 的由来)

1.  **原始图片加载**:
    *   使用 PIL 或 OpenCV 读取图片 (RGB 格式, $H \times W \times 3$)。

2.  **预处理 (Image Processor)**:
    *   **Resize**: 将图片调整为特定分辨率 (如 $336 \times 336$ 或动态分辨率)。
    *   **Normalization**: 减去均值，除以方差。
    *   **ToTensor**: 转换为 PyTorch Tensor。
    *   产出 **`pixel_values`**: 形状通常为 `[Batch, Channels, Height, Width]`。

3.  **视觉编码 (Vision Encoder)**:
    *   这是最关键的一步，发生在 `Qwen3VLModel.get_image_features` 中。
    *   **输入**: `pixel_values`
    *   **Patch Embedding**: 通过卷积层将图片切成小块 (Patches)。
    *   **Transformer Layers**: 经过多层 Vision Transformer (ViT) 处理。
    *   **Pooling/Reshape**: 可能会进行池化或重塑，将 2D 特征图展平为 1D 序列。
    *   **Projection**: 最后通过一个线性层 (Projection Layer) 将特征维度对齐到 LLM 的 Hidden Size（例如 4096）。
    *   **产出**: **`image_tokens`** (Embeddings)。这就是你在 JSON 文件中看到的巨大浮点数数组。

---

## 3. 最终融合 (Fusion)

在模型的 `forward` 函数中：

1.  模型先将 `input_ids` (整数) 转换为初步的 Embeddings。此时，`151655` 对应的 Embedding 是毫无意义的初始化向量。
2.  模型根据 `input_ids` 找到 `151655` 所在的位置索引。
3.  模型将 **`image_tokens`** (来自 Vision Encoder 的真实特征) **直接覆盖/替换** 掉那些位置上的 Embeddings。

**结果**: Transformer 处理的序列中，文本位置是文本特征，图片位置是真实的视觉特征。
