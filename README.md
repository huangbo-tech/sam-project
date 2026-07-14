# SAM 3 基于概念的图像与视频分割

*计算机视觉课程大作业*

---

## 一、项目概述

SAM 3（Segment Anything with Concepts）是 Meta Superintelligence Labs 发布的统一基础模型，用于图像和视频中的可提示分割任务。它可以通过文本、点、框、掩码等提示方式，对目标进行检测、分割和跟踪。

与 SAM 2 相比，SAM 3 的核心提升是：不仅能根据视觉提示完成分割，还可以根据一个简短的开放词汇文本概念，穷尽式地找出图像或视频中所有符合该概念的实例。例如输入 “a player in white” 或 “truck”，模型会尝试检测并分割所有符合描述的对象。

![文档图片 1](计算机视觉+黄博+2025Z8017782003-大作业_assets/image_1.png)

SAM 3 architecture

SAM 3 在 SA-Co 基准测试上达到人类表现的 75% 到 80%。SA-Co 包含约 27 万个独立概念，概念数量是已有同类基准的 50 倍以上。该能力主要来自一个自动化数据引擎：它标注了超过 400 万个独立概念，构建出目前规模最大的高质量开放词汇分割数据集之一。

SAM 3 的模型结构中引入了 presence token，用于增强模型区分相近文本提示的能力，例如区分 “a player in white” 和 “a player in red”。同时，它采用检测器和跟踪器解耦的设计，降低任务之间的干扰，使模型可以更高效地随数据规模扩展。

![文档图片 2](计算机视觉+黄博+2025Z8017782003-大作业_assets/image_2.png)

![文档图片 3](计算机视觉+黄博+2025Z8017782003-大作业_assets/image_3.png)

项目相关链接：

- 论文：[SAM 3: Segment Anything with Concepts](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)

- 项目页：[https://ai.meta.com/sam3](https://ai.meta.com/sam3)

- 在线演示：[https://segment-anything.com/](https://segment-anything.com/)

- 官方博客：[Segment Anything Model 3](https://ai.meta.com/blog/segment-anything-model-3/)

- 代码仓库：[https://github.com/facebookresearch/sam3](https://github.com/facebookresearch/sam3)

## 二、环境安装

### 1. 创建 Conda 环境

```bash
conda create -n sam3 python=3.12
conda deactivate
conda activate sam3
```

### 2. 安装支持 CUDA 的 PyTorch

```bash
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

如果本机没有 CUDA GPU，可以安装 CPU 版本 PyTorch，但推理速度会明显降低，视频任务可能较慢。

### 3. 克隆仓库并安装项目

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

如果已经在当前本地目录 sam3-main 中，则进入该目录后直接安装：

```bash
cd sam3-main
pip install -e .
```

### 4. 安装 Notebook 和开发依赖

运行示例 Notebook 需要安装额外依赖：

```bash
pip install -e ".[notebooks]"
```

如果需要训练、开发、格式化或测试相关功能，可以安装：

```bash
pip install -e ".[train,dev]"
```

开发环境中常用格式化命令：

```bash
ufmt format .
```

## 三、模型权重准备

使用 SAM 3 前，需要获取模型 checkpoint。官方 README 提示需要先在 Hugging Face 的 [facebook/sam3](https://huggingface.co/facebook/sam3) 页面申请访问权限。通过后，需要登录 Hugging Face 才能自动下载权重。

登录方式示例：

```bash
hf auth login
```

登录前需要在 Hugging Face 账户中创建 access token。

本地仓库中已经存在 sam3.pt 权重文件时，也可以直接使用本地权重，避免每次从 Hugging Face 下载。构建模型时设置：

```text
checkpoint_path="sam3.pt"
load_from_HF=False
```

## 四、图片使用方法

SAM 3 可以对单张图片进行开放词汇实例分割。输入是一张图片和一个文本提示，输出包括：

- masks：目标实例的分割掩码

- boxes：目标实例的边界框

- scores：检测置信度

### 1. 基础图片推理代码

```python
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# 加载模型。若使用 Hugging Face 自动下载权重，可直接调用 build_sam3_image_model()
model = build_sam3_image_model()
processor = Sam3Processor(model)

# 读取图片
image = Image.open("<YOUR_IMAGE_PATH.jpg>")

# 设置图片状态
inference_state = processor.set_image(image)

# 使用文本提示进行分割，例如 "dog"、"truck"、"a player in white"
output = processor.set_text_prompt(
    state=inference_state,
    prompt="<YOUR_TEXT_PROMPT>",
)

# 读取输出结果
masks = output["masks"]
boxes = output["boxes"]
scores = output["scores"]
```

### 2. 使用本地权重进行图片推理

如果当前目录下已有 sam3.pt，可以使用以下写法：

```python
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

device = "cuda" if torch.cuda.is_available() else "cpu"

model = build_sam3_image_model(
    device=device,
    checkpoint_path="sam3.pt",
    load_from_HF=False,
)

processor = Sam3Processor(
    model,
    device=device,
    confidence_threshold=0.8,
)

image = Image.open("assets/images/truck.jpg").convert("RGB")
state = processor.set_image(image)
output = processor.set_text_prompt(state=state, prompt="truck")

masks = output["masks"]
boxes = output["boxes"]
scores = output["scores"]

print("检测数量：", len(scores))
print("边界框：", boxes)
print("置信度：", scores)
```

### 3. 运行本仓库中的图片烟雾测试脚本

本地仓库提供了 run_sam3_smoke.py，可用于快速验证图片推理是否可用：

```bash
python run_sam3_smoke.py --image assets/images/truck.jpg --prompt truck --checkpoint sam3.pt --threshold 0.8
```

运行后会在 outputs/smoke 中生成：

- sam3_smoke_overlay.jpg：叠加分割掩码和边界框后的可视化结果

- sam3_smoke_result.json：包含图片路径、提示词、检测数量、边界框和置信度的 JSON 结果

## 五、视频使用方法

SAM 3 也可以对视频进行文本提示分割和目标跟踪。输入可以是：

- 一个 MP4 视频文件

- 一个按帧排列的 JPEG 图片文件夹

视频推理采用会话式接口：先创建视频会话，再向指定帧添加文本提示，模型随后返回对应目标在视频中的输出结果。

### 1. 基础视频推理代码

```python
from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor()

# video_path 可以是 MP4 文件，也可以是 JPEG 帧文件夹
video_path = "<YOUR_VIDEO_PATH>"

# 1. 创建视频会话
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)

session_id = response["session_id"]

# 2. 在指定帧添加文本提示
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=0,
        text="<YOUR_TEXT_PROMPT>",
    )
)

# 3. 获取视频分割和跟踪结果
output = response["outputs"]
```

### 2. 使用本地权重构建视频预测器

如果不希望从 Hugging Face 自动下载权重，可以传入本地 checkpoint：

```python
from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor(
    checkpoint_path="sam3.pt",
    load_from_HF=False,
)
```

### 3. 视频交互细化

官方示例中还支持在视频上进一步进行交互式细化，例如在某些帧添加点提示，对跟踪结果进行修正。相关内容可查看：

```text
examples/sam3_video_predictor_example.ipynb
examples/sam3_for_sam2_video_task_example.ipynb
```

## 六、示例 Notebook

examples 目录包含多种使用方式示例：

- sam3_image_predictor_example.ipynb：图像文本提示和框提示分割示例。

- sam3_video_predictor_example.ipynb：视频文本提示分割，以及使用点进行交互式细化。

- sam3_image_batched_inference.ipynb：图像批量推理示例。

- sam3_agent.ipynb：使用 SAM 3 Agent 处理复杂文本提示。

- saco_gold_silver_vis_example.ipynb：SA-Co 图像评估集可视化示例。

- saco_veval_vis_example.ipynb：SA-Co 视频评估集可视化示例。

- saco_gold_silver_eval_example.ipynb：SA-Co Gold/Silver 图像评估示例。

- saco_veval_eval_example.ipynb：SA-Co 视频评估示例。

- sam3_for_sam1_task_example.ipynb：使用 SAM 3 完成类似 SAM 1 的图像交互式实例分割任务。

- sam3_for_sam2_video_task_example.ipynb：使用 SAM 3 完成类似 SAM 2 的视频交互式分割任务。

- sam3_image_interactive.ipynb：图像交互式分割示例。

运行 Notebook 前先安装依赖：

```bash
pip install -e ".[notebooks]"
```

启动示例：

```bash
jupyter notebook examples/sam3_image_predictor_example.ipynb
```

## 七、模型结构

SAM 3 由检测器和跟踪器组成，两者共享视觉编码器。模型总参数量约为 848M。

检测器采用基于 DETR 的结构，可以同时接受文本、几何提示和图像示例作为条件。跟踪器继承了 SAM 2 的 Transformer 编码器-解码器结构，支持视频分割和交互式细化。

这种检测器和跟踪器解耦的设计有两个好处：

- 检测任务和跟踪任务之间的相互干扰更少。

- 模型可以更好地扩展到大规模数据和开放词汇概念。

## 八、图像实验结果

原 README 给出了 SAM 3 在图像实例分割和框检测任务上的结果。主要指标包括：

- cgF1：概念级 F1 指标

- AP：平均精度

- APo：COCO-O 准确率

| 模型 | LVIS 分割 cgF1 | LVIS 分割 AP | SA-Co/Gold 分割 cgF1 | LVIS 检测 cgF1 | LVIS 检测 AP | COCO AP | COCO APo | SA-Co/Gold 检测 cgF1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Human | - | - | 72.8 | - | - | - | - | 74.0 |
| OWLv2* | 29.3 | 43.4 | 24.6 | 30.2 | 45.5 | 46.1 | 23.9 | 24.5 |
| DINO-X | - | 38.5 | 21.3 | - | 52.4 | 56.0 | - | 22.5 |
| Gemini 2.5 | 13.4 | - | 13.0 | 16.1 | - | - | - | 14.4 |
| SAM 3 | 37.2 | 48.5 | 54.1 | 40.6 | 53.6 | 56.4 | 55.7 | 55.7 |

注：OWLv2 部分使用 LVIS 训练；APo 指 COCO-O 准确率。

## 九、视频实验结果

原 README 还给出了 SAM 3 在多个视频测试集上的结果：

| 模型 | SA-V cgF1 | SA-V pHOTA | YT-Temporal-1B cgF1 | YT-Temporal-1B pHOTA | SmartGlasses cgF1 | SmartGlasses pHOTA | LVVIS mAP | BURST HOTA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Human | 53.1 | 70.5 | 71.2 | 78.4 | 58.5 | 72.3 | - | - |
| SAM 3 | 30.3 | 58.0 | 50.8 | 69.9 | 36.4 | 63.6 | 36.3 | 44.5 |

这些结果说明 SAM 3 不仅能完成静态图像中的开放词汇实例分割，也可以在视频中对概念目标进行跟踪和分割。

## 十、SA-Co 数据集

SAM 3 同时发布了 SA-Co 数据集和评估基准，包括：

- 图像基准：[SA-Co/Gold](scripts/eval/gold/README.md)

- 图像基准：[SA-Co/Silver](scripts/eval/silver/README.md)

- 视频基准：[SA-Co/VEval](scripts/eval/veval/README.md)

数据集中每张图片或每个视频都配有名词短语标注。每个“图片或视频 + 名词短语”组合都会标注与该短语匹配的实例掩码和唯一目标 ID。如果某个短语在图像或视频中没有对应目标，则作为负样本提示，不包含掩码。

数据集托管地址：

- Hugging Face：

  - [SA-Co/Gold](https://huggingface.co/datasets/facebook/SACo-Gold)

  - [SA-Co/Silver](https://huggingface.co/datasets/facebook/SACo-Silver)

  - [SA-Co/VEval](https://huggingface.co/datasets/facebook/SACo-VEval)

- Roboflow：

  - [SA-Co/Gold](https://universe.roboflow.com/sa-co-gold)

  - [SA-Co/Silver](https://universe.roboflow.com/sa-co-silver)

  - [SA-Co/VEval](https://universe.roboflow.com/sa-co-veval)

![文档图片 4](计算机视觉+黄博+2025Z8017782003-大作业_assets/image_4.jpeg)

SA-Co dataset

## 十一、总结

本项目展示了 SAM 3 在图像和视频开放词汇分割任务中的使用流程。环境配置方面，需要准备 Python、PyTorch、CUDA 和项目依赖；模型使用方面，需要获取或指定 sam3.pt 权重；图片任务中，模型根据文本提示输出掩码、边界框和置信度；视频任务中，模型通过会话接口在指定帧添加文本提示，并进一步完成目标分割和跟踪。

通过 SAM 3，可以较方便地完成计算机视觉课程中的图像分割、目标检测、视频目标跟踪、开放词汇识别和交互式分割等实验任务。
