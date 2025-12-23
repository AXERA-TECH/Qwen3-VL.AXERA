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


至此，整个模型转换完毕。将 ../Qwen3-VL-2B-Instruct--AX650-C128_P1152_CTX2047 上传到爱芯的设备上准备运行。    