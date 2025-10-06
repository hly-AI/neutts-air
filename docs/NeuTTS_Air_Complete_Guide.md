# NeuTTS Air 完整指南 ☁️

> 世界首个超逼真、本地部署的TTS语音语言模型，支持即时语音克隆

## 📋 目录

- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [技术架构](#技术架构)
- [环境搭建](#环境搭建)
- [核心文件分析](#核心文件分析)
- [Perth水印技术](#perth水印技术)
- [运行步骤](#运行步骤)
- [使用示例](#使用示例)
- [故障排除](#故障排除)
- [参考资料](#参考资料)

## 🎯 项目概述

NeuTTS Air 是由 [Neuphonic](http://neuphonic.com/) 开发的世界首个超逼真、本地部署的TTS语音语言模型。它基于0.5B LLM骨干网络，将自然语音合成、实时性能、内置安全性和说话人克隆功能带到您的本地设备。

### 🌟 主要亮点

- **🗣 最佳逼真度** - 产生自然、超逼真的声音，听起来像真人
- **📱 本地部署优化** - 提供GGML格式，可在手机、笔记本电脑甚至树莓派上运行
- **👫 即时语音克隆** - 仅需3秒音频即可创建您自己的说话人
- **🚄 简单架构** - 基于0.5B骨干的LM + 编解码器架构，在速度、大小和质量之间达到最佳平衡

## 🔧 核心特性

### 模型详情

- **骨干网络**: Qwen 0.5B - 轻量级但功能强大的语言模型
- **音频编解码器**: [NeuCodec](https://huggingface.co/neuphonic/neucodec) - 专有神经音频编解码器
- **格式**: GGML格式，支持高效本地推理
- **推理速度**: 在中端设备上实时生成
- **功耗**: 针对移动和嵌入式设备优化

### 支持格式

- **PyTorch模型**: 完整精度模型
- **GGUF量化模型**: 高效推理模型
- **ONNX解码器**: CPU优化解码器

## 🏗 技术架构

### 核心组件

1. **语言模型骨干** - 负责文本理解和生成
2. **音频编解码器** - 负责音频编码和解码
3. **音素化器** - 将文本转换为音素
4. **水印器** - 为生成的音频添加水印

### 工作流程

```
输入文本 → 音素化 → 语言模型推理 → 音频解码 → 水印添加 → 输出音频
```

## 🛠 环境搭建

### 1. 创建Conda环境

```bash
# 创建新的conda环境，使用Python 3.11（项目要求>=3.11）
conda create -n neutts-air python=3.11 -y

# 激活环境
conda activate neutts-air
```

### 2. 安装系统依赖

#### macOS 用户：
```bash
# 安装 espeak（必需的系统依赖）
brew install espeak
```

#### Ubuntu/Debian 用户：
```bash
# 安装 espeak
sudo apt install espeak
```

#### Windows 用户：
需要下载并安装 eSpeak NG，然后设置环境变量：
```powershell
$env:PHONEMIZER_ESPEAK_LIBRARY = "c:\Program Files\eSpeak NG\libespeak-ng.dll"
$env:PHONEMIZER_ESPEAK_PATH = "c:\Program Files\eSpeak NG"
setx PHONEMIZER_ESPEAK_LIBRARY "c:\Program Files\eSpeak NG\libespeak-ng.dll"
setx PHONEMIZER_ESPEAK_PATH "c:\Program Files\eSpeak NG"
```

### 3. 安装Python依赖

```bash
# 确保在激活的conda环境中
conda activate neutts-air

# 安装基础依赖
pip install -r requirements.txt

# 可选：安装 GGUF 模型支持（用于更高效的推理）
pip install llama-cpp-python

# 可选：安装 ONNX 运行时支持（用于 ONNX 解码器）
pip install onnxruntime
```

### 4. macOS 用户特殊配置

由于项目已经包含了 macOS 的 espeak 库路径配置，在 `neuttsair/neutts.py` 文件的第12-13行已经设置好了：

```python
_ESPEAK_LIBRARY = '/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.1.1.48.dylib'
EspeakWrapper.set_library(_ESPEAK_LIBRARY)
```

如果你的 espeak 安装路径不同，需要修改这个路径。

## 📁 核心文件分析

### `neutts.py` - 项目核心

这是整个项目的**核心文件**，包含了所有主要的TTS（文本转语音）功能。

#### 核心类：`NeuTTSAir`

```python
class NeuTTSAir:
    def __init__(
        self,
        backbone_repo="neuphonic/neutts-air",
        backbone_device="cpu",
        codec_repo="neuphonic/neucodec",
        codec_device="cpu",
    ):
```

#### 主要组件和功能

1. **初始化 (`__init__`)**
   - **音素化器**：使用 `EspeakBackend` 将文本转换为音素
   - **骨干模型**：支持多种模型格式（PyTorch、GGUF量化模型）
   - **编解码器**：支持 NeuCodec、DistillNeuCodec、ONNX解码器
   - **水印器**：使用 Perth 水印技术

2. **模型加载**
   - **`_load_backbone()`**：加载语言模型
     - 支持 GGUF 量化模型（通过 llama-cpp-python）
     - 支持标准 PyTorch 模型（通过 transformers）
   - **`_load_codec()`**：加载音频编解码器
     - `neuphonic/neucodec`：标准编解码器
     - `neuphonic/distill-neucodec`：蒸馏版编解码器
     - `neuphonic/neucodec-onnx-decoder`：ONNX解码器

3. **核心推理功能**
   - **`infer()`**：主要的推理方法
     - 输入：文本、参考音频编码、参考文本
     - 输出：生成的语音波形
   - **`encode_reference()`**：将参考音频编码为token序列

4. **文本处理**
   - **`_to_phones()`**：将文本转换为音素表示
   - **`_apply_chat_template()`**：构建聊天模板用于模型推理

5. **推理引擎**
   - **`_infer_torch()`**：使用PyTorch模型进行推理
   - **`_infer_ggml()`**：使用GGML量化模型进行推理
   - **`_decode()`**：将生成的token解码为音频波形

#### 技术特点

1. **多格式支持**：
   - PyTorch模型（完整精度）
   - GGUF量化模型（高效推理）
   - ONNX解码器（CPU优化）

2. **设备灵活性**：
   - CPU/GPU支持
   - 自动设备检测和分配

3. **音素化处理**：
   - 使用espeak进行文本到音素的转换
   - 支持重音和标点符号保留

4. **水印技术**：
   - 自动为生成的音频添加Perth水印

## 🎵 Perth水印技术

### 什么是Perth水印？

**Perth** 是一个**感知阈值水印器**（Perceptual Threshold Watermarker），它是一种**不可感知的音频水印技术**，专门设计用于在音频中嵌入数字信息，同时保持音频质量不受影响。

### 核心特点

#### 1. 不可感知性
- 水印嵌入在**人耳听觉阈值以下**
- 正常聆听时**完全听不到**水印的存在
- 不会影响音频的**音质和听感**

#### 2. 鲁棒性
- 能够**抵抗常见的音频处理**：
  - 音频压缩（MP3、AAC等）
  - 重采样
  - 音量调整
  - 格式转换
  - 噪声添加

#### 3. 神经网络技术
- 使用**Perth-Net Implicit**神经网络方法
- 基于**深度学习**的嵌入和提取算法
- 比传统水印技术更加**智能和稳定**

### 在NeuTTS Air中的应用

```python
# 在neutts.py中的使用
self.watermarker = perth.PerthImplicitWatermarker()

# 自动为生成的音频添加水印
watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=24_000)
```

### 水印的作用

#### 1. 版权保护
- 为生成的音频添加**唯一标识**
- 可以**追踪音频来源**
- 防止**未经授权的复制和分发**

#### 2. 内容认证
- 验证音频的**完整性和真实性**
- 检测音频是否被**篡改**
- 确保音频**未被恶意修改**

#### 3. 责任追踪
- 每个NeuTTS Air生成的音频都包含水印
- 可以**识别音频的生成来源**
- 符合**AI伦理和合规要求**

### 技术原理

#### 1. 感知掩蔽
- 利用**人耳听觉特性**
- 在**听觉阈值以下**嵌入信息
- 确保水印**不可察觉**

#### 2. 频域嵌入
- 在**特定频率**嵌入水印信号
- 选择**人耳不敏感**的频率范围
- 保持**音频频谱平衡**

#### 3. 神经网络优化
- 使用**深度学习**优化嵌入过程
- **自适应调整**水印强度
- 最大化**鲁棒性**和**不可感知性**

### 检测和提取

```python
# 可以检测和提取水印
watermarker = perth.PerthImplicitWatermarker()
watermark = watermarker.get_watermark(watermarked_audio, sample_rate=sr)
print(f"检测到的水印: {watermark}")
```

### 伦理考虑

NeuTTS Air使用Perth水印的主要目的是：

1. **负责任AI**：确保AI生成内容的可追溯性
2. **防止滥用**：防止恶意使用AI生成的语音
3. **合规要求**：满足法律和伦理标准
4. **透明度**：让用户知道音频是AI生成的

## 🚀 运行步骤

### 基础示例运行

```bash
# 使用默认模型
python -m examples.basic_example \
  --input_text "My name is Dave, and um, I'm from London" \
  --ref_audio samples/dave.wav \
  --ref_text samples/dave.txt

# 使用 GGUF 量化模型（更高效）
python -m examples.basic_example \
  --input_text "My name is Dave, and um, I'm from London" \
  --ref_audio samples/dave.wav \
  --ref_text samples/dave.txt \
  --backbone neuphonic/neutts-air-q4-gguf
```

### 使用自定义音频和文本

```bash
python -m examples.basic_example \
  --input_text "你想要合成的文本内容" \
  --ref_audio "你的参考音频文件.wav" \
  --ref_text "参考音频对应的文本内容" \
  --output_path "输出文件名.wav"
```

### ONNX 解码器示例

```bash
# 首先编码参考音频
python -m examples.encode_reference \
  --ref_audio samples/dave.wav \
  --output_path encoded_reference.pt

# 然后使用 ONNX 解码器
python -m examples.onnx_example \
  --input_text "My name is Dave, and um, I'm from London" \
  --ref_codes encoded_reference.pt \
  --ref_text samples/dave.txt
```

### 交互式 Jupyter 示例

```bash
# 启动 Jupyter notebook
jupyter notebook examples/interactive_example.ipynb
```

## 💻 使用示例

### 简单代码块使用

```python
from neuttsair.neutts import NeuTTSAir
import soundfile as sf

# 初始化
tts = NeuTTSAir(
    backbone_repo="neuphonic/neutts-air-q4-gguf", 
    backbone_device="cpu", 
    codec_repo="neuphonic/neucodec", 
    codec_device="cpu"
)

input_text = "My name is Dave, and um, I'm from London."

ref_text = "samples/dave.txt"
ref_audio_path = "samples/dave.wav"

ref_text = open(ref_text, "r").read().strip()
ref_codes = tts.encode_reference(ref_audio_path)

wav = tts.infer(input_text, ref_codes, ref_text)
sf.write("test.wav", wav, 24000)
```

### 高级使用示例

#### GGML 骨干模型示例
```bash
python -m examples.basic_example \
  --input_text "My name is Dave, and um, I'm from London" \
  --ref_audio ./samples/dave.wav \
  --ref_text ./samples/dave.txt \
  --backbone neuphonic/neutts-air-q4-gguf
```

#### ONNX 解码器示例
```bash
python -m examples.onnx_example \
  --input_text "My name is Dave, and um, I'm from London" \
  --ref_codes samples/dave.pt \
  --ref_text samples/dave.txt
```

## 📊 项目依赖说明

- **librosa==0.11.0**: 音频处理
- **neucodec>=0.0.4**: 神经音频编解码器
- **numpy==2.2.6**: 数值计算
- **phonemizer==3.3.0**: 音素化处理
- **soundfile==0.13.1**: 音频文件读写
- **torch==2.8.0**: PyTorch 深度学习框架
- **transformers==4.56.1**: Hugging Face 模型库
- **resemble-perth==1.0.1**: 音频水印技术

## 🎵 参考音频要求

为了获得最佳效果，参考音频应该：

1. **单声道**
2. **16-44 kHz 采样率**
3. **3–15 秒长度**
4. **WAV 格式**
5. **清晰** — 最小化背景噪音
6. **自然连续语音** — 像独白或对话，停顿较少，让模型能有效捕捉语调

### 示例参考文件

您可以在 `samples` 文件夹中找到一些现成的样本：

- `samples/dave.wav`
- `samples/jo.wav`

## 🔧 故障排除

### 常见问题

1. **espeak 安装问题**
   - 确保 espeak 正确安装
   - 检查库路径是否正确（macOS用户）

2. **Python 版本问题**
   - 确保 Python 版本 >= 3.11

3. **依赖包问题**
   - 验证所有依赖包正确安装
   - 检查 PyTorch 版本兼容性

4. **模型下载问题**
   - 确保网络连接正常
   - 检查 HuggingFace 访问权限

### 调试步骤

1. 检查环境变量
2. 验证文件路径
3. 查看错误日志
4. 测试简单示例

## 🌐 Web界面

### HuggingFace Spaces

官方在HuggingFace上提供了一个基于Gradio的web界面：
- **链接**: https://huggingface.co/spaces/neuphonic/neutts-air
- **功能**: 上传参考音频、提供参考文本、输入新文本进行合成

### 本地项目

- **命令行界面**: 通过命令行参数运行
- **Python API**: 通过编程接口使用
- **Jupyter Notebook**: 交互式使用

### 自定义Web界面

基于核心类 `NeuTTSAir`，您可以轻松创建自己的web界面：

```python
# 示例：Flask web应用
from flask import Flask, request, send_file
from neuttsair.neutts import NeuTTSAir
import tempfile
import os

app = Flask(__name__)
tts = NeuTTSAir()

@app.route('/synthesize', methods=['POST'])
def synthesize():
    # 处理音频上传和文本输入
    # 使用 tts.infer() 生成语音
    # 返回生成的音频文件
    pass
```

## 📚 参考资料

### 官方资源

- **GitHub**: https://github.com/neuphonic/neutts-air
- **HuggingFace**: https://huggingface.co/neuphonic/neutts-air
- **HuggingFace Spaces**: https://huggingface.co/spaces/neuphonic/neutts-air
- **YouTube演示**: https://www.youtube.com/watch?v=YAB3hCtu5wE

### 相关技术

- **NeuCodec**: https://huggingface.co/neuphonic/neucodec
- **Perth水印**: https://github.com/resemble-ai/perth
- **espeak**: https://github.com/espeak-ng/espeak-ng
- **Qwen模型**: https://huggingface.co/Qwen

### 模型集合

- **NeuTTS-Air Collection**: https://huggingface.co/collections/neuphonic/neutts-air-68cc14b7033b4c56197ef350

## ⚖️ 免责声明

请不要使用此模型做坏事...请负责任地使用。

## 🤝 贡献

欢迎贡献！请随时提交 Pull Request。

---

*本文档涵盖了NeuTTS Air项目的所有核心知识点和运行步骤。如有问题，请参考官方文档或提交Issue。*
