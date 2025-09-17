# Qianfan-VL: 领域增强通用视觉-语言模型系列

<p align="center">
  <strong>通过持续预训练实现领域能力增强 | 3B到70B参数规模 | 文档理解与OCR能力增强 | 支持思考推理能力</strong>
</p>

<div align="center">

🤗 **[Hugging Face 模型库](https://huggingface.co/baidu)** |
🤖 **[ModelScope 模型库](https://modelscope.cn/organization/baidu-qianfan)** |
📚 **[使用教程 Cookbook](https://github.com/baidubce/qianfan-models-cookbook)** |
📖 **[技术博客](https://baidubce.github.io/Qianfan-VL)** |
📄 **技术报告 [待更新]**

</div>

---

## 模型介绍

Qianfan-VL模型系列是在企业级应用多模态大模型的场景中进行强化的通用多模态大模型，具备基础的通用能力，同时在产业落地的高频场景有深度的优化。通过三大核心功能，精准满足不同场景下的多模态理解需求。

## 核心特性

### 🚀 多尺寸模型
提供3B、8B、70B三种规格的模型，满足从端侧到云端的不同场景需求

### 📝 OCR与文档理解增强
- **全场景OCR识别**：支持手写体、印刷体、场景文字、公式等多种文字识别
- **复杂版面理解**：表格解析、图表理解、文档结构化等能力
- **多语言支持**：中英文及多语言文档处理能力

### 🧠 思考推理能力
8B和70B模型支持思考链（Chain-of-Thought）能力，在数学、推理计算等复杂场景展现卓越表现，可应用于辅助教学、拍照解题、自动判题等场景

## 模型规格

| 模型名称 | 参数量 | 上下文长度 | 支持思考 | 适用场景 | 模型下载 |
|---------|--------|-----------|---------|----------|---------|
| **Qianfan-VL-3B** | 3B | 32k | ❌ | 端上实时场景、OCR文字识别 | 🤗 **[HuggingFace](https://huggingface.co/baidu/Qianfan-VL-3B)** / 🤖 **[ModelScope](https://modelscope.cn/models/baidu-qianfan/Qianfan-VL-3B)** |
| **Qianfan-VL-8B** | 8B | 32k | ✅ | 服务端通用场景、微调优化场景 | 🤗 **[HuggingFace](https://huggingface.co/baidu/Qianfan-VL-8B)** / 🤖 **[ModelScope](https://modelscope.cn/models/baidu-qianfan/Qianfan-VL-8B)** |
| **Qianfan-VL-70B** | 70B | 32k | ✅ | 离线数据合成、复杂推理计算场景 | 🤗 **[HuggingFace](https://huggingface.co/baidu/Qianfan-VL-70B)** / 🤖 **[ModelScope](https://modelscope.cn/models/baidu-qianfan/Qianfan-VL-70B)** |

## 技术优势

### 🚀 多阶段领域增强持续预训练技术
采用创新的四阶段渐进式训练策略，从跨模态对齐到通用知识注入，再到领域增强知识注入和后训练对齐，在保持通用能力的同时显著提升领域专项能力。通过精心设计的数据配比和训练策略，实现了通用与专业能力的良好平衡。

### 🎯 高精度数据合成
构建了覆盖文档识别、数学解题、图表理解、表格识别、公式识别、自然场景OCR等核心任务的多任务数据合成管线。结合传统CV模型和程序化生成方法，通过精细化的管线设计和中间过程数据构造，实现了高质量训练数据的高效生产，显著提升了模型在长尾场景的泛化能力。

### ⚡ 大规模昆仑芯集群并行训练
基于百度自研昆仑芯P800芯片，通过5000+卡的超大规模分布式训练系统完成全部模型规模的训练。采用3D并行训练策略和通信-计算融合技术，实现了90%+的集群扩展效率，3T tokens训练数据的高效处理，展示了国产AI基础设施的成熟能力。

## 性能指标

### 通用能力评测

*注：加粗数值表示该指标在所有模型中排名前两位*

| 基准测试 | Qianfan-VL-3B | Qianfan-VL-8B | Qianfan-VL-70B | Intern3-VL-8B | Intern3-VL-78B | Qwen2.5-VL-7B | Qwen2.5-VL-72B |
|---------|---------------|---------------|----------------|---------------|----------------|---------------|----------------|
| **A-Bench_VAL** | 75.65 | 75.72 | **78.1** | 75.86 | 75.86 | 76.49 | **79.22** |
| **CCBench** | 66.86 | 70.39 | **80.98** | **77.84** | 70.78 | 57.65 | 73.73 |
| **SEEDBench_IMG** | 76.55 | 78.02 | **79.13** | 77.00 | 77.52 | 76.98 | **78.34** |
| **SEEDBench2_Plus** | 67.59 | 70.97 | **73.17** | 69.52 | 68.47 | 70.93 | **73.25** |
| **ScienceQA_TEST** | 95.19 | **97.62** | **98.76** | 97.97 | 97.17 | 85.47 | 92.51 |
| **ScienceQA_VAL** | 93.85 | **97.62** | **98.81** | 97.81 | 95.14 | 83.59 | 91.32 |
| **MMT-Bench_VAL** | 62.23 | 63.22 | **71.06** | 65.17 | 63.67 | 61.40 | **69.49** |
| **MTVQA_TEST** | 26.5 | 30.14 | **32.18** | 30.30 | 27.62 | 29.08 | **31.48** |
| **BLINK** | 49.97 | 56.81 | **59.44** | 55.87 | 51.87 | 54.55 | **63.02** |
| **MMStar** | 57.93 | 64.07 | **69.47** | **68.40** | 66.07 | 61.53 | 66.00 |
| **POPE** | 85.08 | 86.06 | 88.97 | **90.59** | 88.87 | 85.97 | 83.35 |
| **RefCOCO (Avg)** | 85.94 | 89.37 | **91.01** | 89.65 | **91.40** | 86.56 | 90.25 |

### OCR与文档理解能力

| 基准测试 | Qianfan-VL-3B | Qianfan-VL-8B | Qianfan-VL-70B | Qwen2.5-VL-3B | Intern3-VL-8B | Intern3-VL-78B | Qwen2.5-VL-7B | Qwen2.5-VL-72B |
|---------|---------------|---------------|----------------|---------------|---------------|----------------|---------------|----------------|
| **OCRBench** | 831 | 854 | 873 | 810 | **881** | 847 | **883** | 874 |
| **AI2D_TEST** | 81.38 | **85.07** | **87.73** | 77.07 | **85.07** | 83.55 | 80.472 | 83.84 |
| **OCRVQA_TEST** | **66.15** | **68.98** | **74.06** | 69.24 | 39.03 | 35.58 | **71.02** | 66.8 |
| **TextVQA_VAL** | 80.11 | 82.13 | **84.48** | 79.09 | 82.15 | 83.52 | **84.962** | 83.26 |
| **DocVQA_VAL** | 90.85 | 93.54 | 94.75 | 92.71 | 92.04 | 83.82 | **94.91** | **95.75** |
| **ChartQA_TEST** | 81.79 | **87.72** | **89.6** | 83.4 | 85.76 | 82.04 | 86.68 | 87.16 |

### 数学推理能力

| 基准测试 | Qianfan-VL-8B | Qianfan-VL-70B | Intern3-VL-8B | Intern3-VL-78B | Qwen2.5-VL-7B | Qwen2.5-VL-72B |
|---------|---------------|----------------|---------------|----------------|---------------|----------------|
| **Mathvista-mini** | **69.19** | **78.6** | 69.5 | 71.1 | 69.5 | 70.1 |
| **Mathvision** | **32.82** | **50.29** | 21.48 | 33.48 | 29.61 | 34.8 |
| **Mathverse** | **48.4** | **61.04** | 30.96 | 43.32 | 43.68 | 49.26 |
| **ChartQA Pro** | **50.41** | **52** | 19.38 | 47.92 | 37.32 | 44.43 |
| **HallusionBench** | **51.72** | **54.52** | 49.7 | 40.5 | 49.2 | 40.2 |
| **InHouse Dataset A** | **59.87** | **71.78** | 26 | 43.40 | 40.64 | 41.47 |
| **InHouse Dataset B** | **61.33** | **75.6** | 26.81 | 39.7 | 36.25 | 42.65 |

## 快速开始

### 安装依赖

```bash
pip install transformers torch torchvision pillow
```

### 使用 Transformers

```python
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # 计算现有图像的宽高比
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # 找到最接近目标的宽高比
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # 计算目标宽度和高度
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # 调整图像大小
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # 分割图像
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# 加载模型
MODEL_PATH = "Baidu/Qianfan-VL-8B"  # 或选择 Qianfan-VL-3B, Qianfan-VL-70B
model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 加载并处理图像
pixel_values = load_image("./example/scene_ocr.png").to(torch.bfloat16)

# 推理
prompt = "<image>请识别图中所有文字"
with torch.no_grad():
    response = model.chat(
        tokenizer,
        pixel_values=pixel_values,
        question=prompt,
        generation_config={"max_new_tokens": 512},
        verbose=False
    )
print(response)
```

### 使用 vLLM

您可以使用 vLLM 的官方 Docker 镜像部署 Qianfan-VL，实现高性能推理和 OpenAI 兼容的 API：

#### 启动 vLLM 服务

```bash
docker run -d --name qianfan-vl \
  --gpus all \
  -v /path/to/Qianfan-VL-8B:/model \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model /model \
  --served-model-name qianfan-vl \
  --trust-remote-code \
  --hf-overrides '{"architectures":["InternVLChatModel"],"model_type":"internvl_chat"}'
```

#### 调用 API

```bash
curl 'http://127.0.0.1:8000/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "qianfan-vl",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": "https://example.com/image.jpg"
            }
          },
          {
            "type": "text",
            "text": "<image>请识别图中所有文字"
          }
        ]
      }
    ]
  }'
```

更多使用示例请参考 [Cookbook](https://github.com/baidubce/qianfan-models-cookbook/blob/main/qianfan-vl/qianfan_vl_example.ipynb)

## 引用

如果您在研究中使用了Qianfan-VL，请引用：

```bibtex
@misc{qianfan-vl-2025,
  title={Qianfan-VL: Domain-Enhanced General Vision-Language Model Series},
  author={Baidu Qianfan Team},
  year={2025},
  publisher={Baidu AI Cloud},
  howpublished={\url{https://github.com/baidubce/Qianfan-VL}}
}
```

## 许可证

本项目遵循 Apache 2.0 许可证。详见 [LICENSE](LICENSE) 文件。

## 联系我们

- 官方网站: [百度智能云千帆](https://qianfan.cloud.baidu.com)
- 技术支持: qianfan-support@baidu.com
- GitHub Issues: [提交问题](https://github.com/baidubce/Qianfan-VL/issues)

---

<p align="center">
  <strong>百度智能云千帆大模型平台 | 2025</strong>
</p>