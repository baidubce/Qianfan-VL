<div align="right">
  <a href="README_CN.md">简体中文</a> | <b>English</b>
</div>

<h1 align="center">Qianfan-VL</h1>

<p align="center">
  <strong>Domain-Enhanced Multimodal Understanding Model</strong><br>
  <strong>3B to 70B Parameters</strong><br>
  <strong>Document Understanding & OCR Enhancement</strong><br>
  <strong>Chain-of-Thought Support</strong>
</p>

<div align="center">

🤗 **[Hugging Face Models](https://huggingface.co/baidu)** |
🤖 **[ModelScope Models](https://modelscope.cn/organization/baidu-qianfan)** |
<a href="https://console.bce.baidu.com/qianfan/modelcenter/model/buildIn/list" style="vertical-align:middle;"><img src="docs/images/ACG.png" alt="ModelBuilder" width="16" height="16" style="vertical-align:middle;"/> **ModelBuilder**</a>

📚 **[Cookbook](https://github.com/baidubce/qianfan-models-cookbook)** |
📖 **[Tech Blog](https://baidubce.github.io/Qianfan-VL)** |
📄 **[Tech Report](https://github.com/baidubce/Qianfan-VL/blob/main/docs/qianfan_vl_report_comp.pdf)**

</div>

---

## Introduction

Qianfan-VL model series is a general-purpose multimodal model enhanced for enterprise-level multimodal applications. It possesses fundamental general capabilities while offering deep optimization for high-frequency industrial deployment scenarios. Through three core functions, it precisely meets multimodal understanding needs in different scenarios.

## Key Features

### 🚀 Multi-Size Models
Provides 3B, 8B, and 70B model variants to meet different scenario requirements from edge to cloud

### 📝 OCR & Document Understanding Enhancement
- **Full-scenario OCR recognition**: Supports handwriting, printed text, scene text, formulas, and more
- **Complex layout understanding**: Table parsing, chart understanding, document structuring capabilities
- **Multi-language support**: Chinese, English, and multilingual document processing

### 🧠 Chain-of-Thought Capability
8B and 70B models support Chain-of-Thought capability, demonstrating excellent performance in complex scenarios like mathematics and reasoning computation, applicable to teaching assistance, photo problem-solving, automatic grading, and more

## Model Specifications

| Model Name | Parameters | Context Length | CoT Support | Application Scenarios | Model Download |
|---------|--------|-----------|---------|----------|---------|
| **Qianfan-VL-3B** | 3B | 32k | ❌ | Edge real-time scenarios, OCR text recognition | 🤗 **[HuggingFace](https://huggingface.co/baidu/Qianfan-VL-3B)** / 🤖 **[ModelScope](https://modelscope.cn/models/baidu-qianfan/Qianfan-VL-3B)** |
| **Qianfan-VL-8B** | 8B | 32k | ✅ | Server-side general scenarios, fine-tuning optimization | 🤗 **[HuggingFace](https://huggingface.co/baidu/Qianfan-VL-8B)** / 🤖 **[ModelScope](https://modelscope.cn/models/baidu-qianfan/Qianfan-VL-8B)** |
| **Qianfan-VL-70B** | 70B | 32k | ✅ | Offline data synthesis, complex reasoning computation | 🤗 **[HuggingFace](https://huggingface.co/baidu/Qianfan-VL-70B)** / 🤖 **[ModelScope](https://modelscope.cn/models/baidu-qianfan/Qianfan-VL-70B)** |

## Technical Advantages

### 🚀 Multi-Stage Domain Enhancement Continuous Pre-training
Adopts an innovative four-stage progressive training strategy, from cross-modal alignment to general knowledge injection, then domain-enhanced knowledge injection and post-training alignment, significantly improving domain-specific capabilities while maintaining general abilities. Through carefully designed data ratios and training strategies, it achieves a good balance between general and professional capabilities.

### 🎯 High-Precision Data Synthesis
Constructs multi-task data synthesis pipelines covering core tasks such as document recognition, mathematical problem-solving, chart understanding, table recognition, formula recognition, and natural scene OCR. By combining traditional CV models with programmatic generation methods, through refined pipeline design and intermediate process data construction, it achieves efficient production of high-quality training data, significantly improving model generalization in long-tail scenarios.

### ⚡ Large-Scale Kunlun Chip Cluster Parallel Training
Based on Baidu's self-developed Kunlun P800 chips, completed training of all model scales through a 5000+ chip ultra-large-scale distributed training system. Using 3D parallel training strategy and communication-computation fusion technology, achieved 90%+ cluster scaling efficiency and efficient processing of 3T tokens training data, demonstrating the mature capabilities of domestic AI infrastructure.

## Performance Metrics

### General Capability Evaluation

*Note: Bold values indicate top-2 rankings among all models*

| Benchmark | Qianfan-VL-3B | Qianfan-VL-8B | Qianfan-VL-70B | InternVL3-8B | InternVL3-78B | Qwen2.5-VL-7B | Qwen2.5-VL-72B |
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
| **POPE** | 85.08 | 86.06 | **88.97** | **90.59** | 88.87 | 85.97 | 83.35 |
| **RefCOCO (Avg)** | 85.94 | 89.37 | **91.01** | 89.65 | **91.40** | 86.56 | 90.25 |

### OCR & Document Understanding

| Benchmark | Qianfan-VL-3B | Qianfan-VL-8B | Qianfan-VL-70B | InternVL3-8B | InternVL3-78B | Qwen2.5-VL-3B | Qwen2.5-VL-7B | Qwen2.5-VL-72B |
|---------|---------------|---------------|----------------|---------------|----------------|---------------|---------------|----------------|
| **OCRBench** | 831 | 854 | 873 | **881** | 847 | 810 | **883** | 874 |
| **AI2D_TEST** | 81.38 | **85.07** | **87.73** | **85.07** | 83.55 | 77.07 | 80.47 | 83.84 |
| **OCRVQA_TEST** | 66.15 | 68.98 | **74.06** | 39.03 | 35.58 | 69.24 | **71.02** | 66.80 |
| **TextVQA_VAL** | 80.11 | 82.13 | **84.48** | 82.15 | 83.52 | 79.09 | **84.96** | 83.26 |
| **DocVQA_VAL** | 90.85 | 93.54 | 94.75 | 92.04 | 83.82 | 92.71 | **94.91** | **95.75** |
| **ChartQA_TEST** | 81.79 | **87.72** | **89.6** | 85.76 | 82.04 | 83.4 | 86.68 | 87.16 |

### Mathematical Reasoning

| Benchmark | Qianfan-VL-8B | Qianfan-VL-70B | InternVL3-8B | InternVL3-78B | Qwen2.5-VL-7B | Qwen2.5-VL-72B |
|---------|---------------|----------------|---------------|----------------|---------------|----------------|
| **MathVista-mini** | 69.19 | **78.6** | 69.5 | 71.1 | 69.5 | 70.1 |
| **MathVision** | 32.82 | **50.29** | 21.48 | 33.48 | 29.61 | 34.8 |
| **MathVerse** | 48.4 | **61.04** | 30.96 | 43.32 | 43.68 | 49.26 |
| **ChartQA Pro** | 50.41 | **52** | 19.38 | 47.92 | 37.32 | 44.43 |
| **HallusionBench** | 51.72 | **54.52** | 49.7 | 40.5 | 49.2 | 40.2 |
| **InHouse Dataset A** | 59.87 | **71.78** | 26 | 43.40 | 40.64 | 41.47 |
| **InHouse Dataset B** | 61.33 | **75.6** | 26.81 | 39.7 | 36.25 | 42.65 |

## Quick Start

### Installation

```bash
pip install transformers torch torchvision pillow
```

### Using Transformers

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

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
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

# Load model
MODEL_PATH = "Baidu/Qianfan-VL-8B"  # or Qianfan-VL-3B, Qianfan-VL-70B
model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Load and process image
pixel_values = load_image("./example/scene_ocr.png").to(torch.bfloat16)

# Inference
prompt = "<image>Please recognize all text in the image"
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

### Using vLLM

You can deploy Qianfan-VL using vLLM's official Docker image for high-performance inference with an OpenAI-compatible API:

#### Start vLLM Service

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

#### Call the API

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
            "text": "<image>Please recognize all text in the image"
          }
        ]
      }
    ]
  }'
```

For more examples, please refer to [Cookbook](https://github.com/baidubce/qianfan-models-cookbook/blob/main/qianfan-vl/qianfan_vl_example.ipynb)

## Citation

If you use Qianfan-VL in your research, please cite:

```bibtex
@misc{qianfan-vl-2025,
  title={Qianfan-VL: Domain-Enhanced Universal Vision-Language Models},
  author={Baidu Qianfan Team},
  year={2025},
  publisher={Baidu AI Cloud},
  howpublished={\url{https://github.com/baidubce/Qianfan-VL}}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact Us

- GitHub Issues: [Submit Issue](https://github.com/baidubce/Qianfan-VL/issues)

---

<p align="center">
  <strong>Baidu AI Cloud Qianfan | 2025</strong>
</p>
