# Qwen3-VL-2B-Instruct 模型转换
这个模型分为 Vision Encoder 和 Language Model 两部分，分别进行转换。

## 一、转换 Vision Encoder 

导出 Vision Encoder 为 onnx，然后通过 `pulsar2 build` 转换为 axmodel模型，

### 1. 创建虚拟环境

```
conda create -n qwen3_vl python=3.12 -y
conda activate qwen3_vl
```

### 2. 安装依赖

```
pip install -r requirements.txt
```

### 3. 导出模型（PyTorch -> ONNX）

在导出onnx之前需要先下从 huggingface 或 model scope 下载模型。这里假设模型的保存目录是 `../Qwen/Qwen3-VL-2B-Instruct/`。    

可以执行`bash export.sh`直接导出模型，以下是详细步骤。  

1). 运行模型，保存导出onnx需要的参数
```
python run_image.py ../Qwen/Qwen3-VL-2B-Instruct/
```
这里会保存 `hidden_states`, `pos_embeds`, `position_embeddings`。  
其中，`pos_embeds`, `position_embeddings`只和图像尺寸相关。所以如果模型的输入尺寸固定，它们两个可以固定到onnx模型中。

2). 导出onnx模型
和模型原始输入不同的是，这里为了让模型使用UINT8输入，特意将`Qwen2VLImageProcessor` 编排过的 image patches 又转换成了图片的格式（具体代码在[preprocess.py](preprocess.py)里面可以看到）。  

```
python export.py ../Qwen/Qwen3-VL-2B-Instruct/
```
这一步会生成 `Qwen3-VL-2B-Instruct_vision.onnx`。

3). 对onnx模型进行simplify 
```
conda create -n py39 python=3.9 -y 
conda activate py39
pip install -r requirements_onnxsim.txt
python sim.py Qwen3-VL-2B-Instruct_vision.onnx
```

4). 测试onnx模型

```
python run_image_onnx.py ../Qwen/Qwen3-VL-2B-Instruct/
```
这一步会用onnx模型替换 vision encoder 模块进行推理。

### 4.转换模型（ONNX -> Axera）

使用模型转换工具 `Pulsar2` 将 ONNX 模型转换成适用于 Axera 的 NPU 运行的模型文件格式 `.axmodel`，通常情况下需要经过以下两个步骤：

- 生成适用于该模型的 PTQ 量化校准数据集
- 使用 `Pulsar2 build` 命令集进行模型转换（PTQ 量化、编译），更详细的使用说明请参考 [AXera Pulsar2 工具链指导手册](https://pulsar2-docs.readthedocs.io/zh-cn/latest/index.html)

1). 生成量化数据集  
这里将图片按照patch编排后，重新保存为图片形式，和onnx模型的输入一致  
```
python get_image_calib.py
cd calib_img
tar -cvf hidden_states.tar *.jpg
```

2). 模型转换

* 修改配置文件
 
检查`config.json` 中 `calibration_dataset` 字段，将该字段配置的路径改为上一步下载的量化数据集存放路径  

* Pulsar2 build

参考命令如下：
build_VE.sh
```
pulsar2 build --input Qwen3-VL-2B-Instruct_vision.onnx \
                --config config.json \
                --output_dir build-output-image-2b \
                --output_name Qwen3-VL-2B-Instruct_vision.axmodel \
                --target_hardware AX650 \
                --compiler.check 0
```
编译完成后将文件`build-output/Qwen3-VL-2B-Instruct_vision.axmodel` 上传到爱芯的设备上.

## 二、转换 Language Model  

### 1. 转换Language Model  
执行命令
build_llm.sh
```
INPUT_DIR=../Qwen/Qwen3-VL-2B-Instruct
OUTPUT_DIR=../Qwen3-VL-2B-Instruct--AX650-C128_P1152_CTX2047
pulsar2 llm_build --input_path $INPUT_DIR \
                --output_path  $OUTPUT_DIR \
                --kv_cache_len 2047 \
                --hidden_state_type bf16 \
                --prefill_len 128 \
                --last_kv_cache_len 128 \
                --last_kv_cache_len 256 \
                --last_kv_cache_len 384 \
                --last_kv_cache_len 512 \
                --last_kv_cache_len 640 \
                --last_kv_cache_len 768 \
                --last_kv_cache_len 896 \
                --last_kv_cache_len 1024 \
                --last_kv_cache_len 1152 \
                --chip AX650 \
                --parallel 8


./tools/embed_process.sh $INPUT_DIR $OUTPUT_DIR
```
其中 `last_kv_cache_len` 的最大值就是 `prefill`阶段的最大token数，请根据实际情况设置这个值。
`parallel` 会启动多进程编译，请根据您的计算机性能设置。

至此，整个模型转换完毕。将 ../Qwen3-VL-2B-Instruct--AX650-C128_P1152_CTX2047 上传到爱芯的设备上准备运行。

## 三、转换带多个 LoRA 的 Qwen3-VL-4B 文本段

本节用于编译 `Qwen/Qwen3-VL-4B-Instruct` 的文本段，并在同一套 AXModel
图上注册多个 PEFT LoRA adapter。LoRA adapter 只修改文本 decoder 的投影层，
不包含 Vision Encoder；Vision Encoder 仍按本文件第一节单独导出和编译。

### 1. 输入目录约定

下面的目录名只是示例，均为相对于 `model_convert/` 的路径，可以通过环境变量
覆盖：

```text
../Qwen/Qwen3-VL-4B-Instruct/       # 原始 Hugging Face 基座模型
../Qwen/qwen3-vl-lora-chartqa/      # LoRA adapter A
../Qwen/qwen3-vl-lora-design/       # LoRA adapter B
```

示例中的两个 adapter 分别来自：

- [nugunaai/Qwen3-VL-4B-ChartQA-lora](https://huggingface.co/nugunaai/Qwen3-VL-4B-ChartQA-lora)
- [raginigupta6/qwen3-vl-4b-design-copilot-grpo](https://huggingface.co/raginigupta6/qwen3-vl-4b-design-copilot-grpo)

它们分别以 ChartQA 和 Design CoPilot 为训练/发布意图。本节只说明其编译和
运行时动态切换方式，不代表已经验证这两个 adapter 的领域任务精度。

每个 adapter 目录必须包含 `adapter_config.json` 和
`adapter_model.safetensors`。多个 adapter 必须共享同一个编译契约：

- 基座模型均为 `Qwen/Qwen3-VL-4B-Instruct`；
- 固定 rank，且覆盖基座的全部文本层；
- 目标模块必须是 `q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、
  `up_proj`、`down_proj`；
- A/B 矩阵形状必须与 Qwen3-VL-4B 文本配置一致；
- `bias=none`，不使用 DoRA、RS-LoRA、QALoRA 或额外可训练模块；
- 源 A/B 权重为 F32 或 BF16，编译产物统一打包为运行时 BF16。

### 2. 编译命令

先进入本目录，并确认当前 `pulsar2` 构建包含 `llm_build2` 的
`--lora_adapter_path` 支持。命令可以直接写入环境变量，也可以只修改四个目录
变量，不需要改脚本：

```bash
MODEL_DIR=../Qwen/Qwen3-VL-4B-Instruct \
OUTPUT_DIR=../Qwen/Qwen3-VL-4B-Instruct-LoRA-AX650-P4K-C6K \
ADAPTER_CHARTQA_DIR=../Qwen/qwen3-vl-lora-chartqa \
ADAPTER_DESIGN_DIR=../Qwen/qwen3-vl-lora-design \
bash build_llm_lora.sh
```

脚本实际执行的核心命令如下；`--lora_adapter_path` 可以重复传入，每个值对应
一个 adapter：

```bash
pulsar2 llm_build2 \
    --input_path "$MODEL_DIR" \
    --output_path "$OUTPUT_DIR" \
    --hidden_state_type bf16 \
    --weight_type s8 \
    --post_weight_type s8 \
    --prefill_len 4096 \
    --prefill_step_size 256 \
    --max_context 6144 \
    --decode_step_size -1 \
    --chip AX650 \
    --parallel 8 \
    --tensor_parallel_size 0 \
    -c 0 \
    --lora_adapter_path "$ADAPTER_CHARTQA_DIR" \
    --lora_adapter_path "$ADAPTER_DESIGN_DIR"
```

参数含义：`--prefill_len 4096` 是总 prefill 容量，
`--prefill_step_size 256` 是每个 prefill 子图的 chunk 大小，
`--max_context 6144` 是最大 decode attention context，
`--decode_step_size -1` 生成单个 decode 子图，`-c 0` 关闭编译阶段的
simulator check，`--tensor_parallel_size 0` 表示非 tensor-parallel 编译。
LoRA matrix-input 当前只支持 AX650、BF16 hidden state 和非 tensor-parallel
配置。

### 3. 提取 embedding 权重

`llm_build2` 完成后，脚本会调用已有的 `tools/embed_process.sh`，从基座模型提取
embedding 并生成运行时需要的 BF16 文件：

```bash
./tools/embed_process.sh "$MODEL_DIR" "$OUTPUT_DIR"
```

最终输出目录应包含 36 个 `qwen3_vl_text_p256_l*_together.axmodel`、一个
`qwen3_vl_text_post.axmodel`、`model.embed_tokens.weight.bfloat16.bin`，以及：

```text
OUTPUT_DIR/lora/<adapter-id>/layer_00.bf16.bin ... layer_35.bf16.bin
OUTPUT_DIR/lora/<adapter-id>/manifest.json
OUTPUT_DIR/lora/<adapter-id>/source_adapter_config.json
```

`<adapter-id>` 取 adapter 目录名。运行时可使用这些目录中的 task ID 动态选择
adapter；编译阶段只需把所有要支持的 adapter 通过重复的
`--lora_adapter_path` 一起传入。当前 runtime 的 active adapter 是进程级状态，
不同 task 的请求需要串行发送。
