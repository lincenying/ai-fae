[当前文档访问路径](https://ai-fae.readthedocs.io/zh-cn/latest/ModelLink介绍.html)

# ModelLink训练
[ModelLink](https://gitee.com/ascend/ModelLink) 基于Ascend PyTorch框架，通过模型并行与数据并行来训练大语言模型，旨在为华为昇腾芯片上提供端到端的大语言模型方案, 包含模型，算法，以及下游任务。  
当前ModelLink支撑大模型使用功能:

- 制作预训练数据集/制作指令微调数据集
- 预训练/全参微调/低参微调
- 流式推理/人机对话
- 评估基线数据集
- 加速算法/融合算子/并行策略
- 基于昇腾芯片采集Profiling数据
- Huggingface与Megatron-LM权重转换
- 基于昇腾芯片的确定性计算功能
- 基于昇腾芯片的高可用特性

## 1. 权重下载

权重可以基于网页直接下载，也可以基于命令行下载，或者使用魔搭

### 1.1 huggingface
huggingface: https://huggingface.co 
huggingface国内镜像: https://hf-mirror.com/models

### 1.2 命令行下载
```bash
mkdir ./model_from_hf/llama-2-7b-hf/
cd ./model_from_hf/llama-2-7b-hf/
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/config.json
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/generation_config.json
wget https://huggingface.co/daryl149/llama-2-7b-hf/resolve/main/pytorch_model-00001-of-00002.bin
...
```

### 1.3 魔搭下载
```bash
pip install modelscope
modelscope download --model qwen/Qwen2-Math-72B-Instruct --local_dir ./qwen-2-math-72b-instruct/

```

## 2. 权重转换

### 2.1 Huggingface权重转换到Megatron-Legacy

```shell
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python tools/checkpoint/convert_ckpt.py \
    --model-type GPT \
    --loader llama2_hf \
    --saver megatron \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 4 \
    --num-layer-list 8,8,8,8 \
    --load-dir ./model_from_hf/llama2-hf/ \
    --save-dir ./model_weights/llama2-legacy/ \
    --tokenizer-model ./model_from_hf/llama2-hf/tokenizer.model
```

【--target-tensor-parallel-size】 指明需要切分的TP数量，默认为1

【--target-pipeline-parallel-size】 指明需要切分的PP数量，默认为1

【--num-layer-list】 可选参数，支持动态PP划分，通过列表指定每个PP Stage的层数

【--num-layers-per-virtual-pipeline-stage】 可选参数，支持VPP划分，指定VPP的每个Stage层数，默认为None

【--tokenizer-model】 需要指明到具体的分词器模型文件，如 tokenizer.model、tokenizer.json、qwen.tiktoken、None等，具体取决于huggingface中词表文件的格式形式

【--params-dtype】 指定权重转换后的权重精度模式，默认为fp16，如果源格式文件为bf16，则需要对应设置为bf16，影响推理或评估结果

### 2.2 Megatron-Legacy权重转换到Huggingface

```bash
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python tools/checkpoint/convert_ckpt.py \
    --model-type GPT \
    --loader megatron \
    --saver megatron \
    --save-model-type save_huggingface_llama \
    --load-dir ./model_weights/llama2-legacy/ \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --save-dir ./model_from_hf/llama2-70b-hf/     # <-- 需要填入原始HF模型路径，新权重会存于./model_from_hf/llama2-hf/mg2hg/
```

### 2.3 lora权重与base权重合并

在上述权重转换命令中，加入如下参数可以将训练的 lora 权重与base进行融合。

```bash
--lora-load ${CHECKPOINT_LORA}  \
--lora-r 16 \
--lora-alpha 32 \
--lora-target-modules query_key_value dense dense_h_to_4h dense_4h_to_h \
```

使用lora权重的评估脚本命名风格及启动方法为：

```bash
bash examples/llama2/evaluate_llama2_7B_lora_ptd.sh
```

---

## 3. 数据处理

### 3.1 数据集下载

从Huggingface等网站下载开源数据集，保存到ModelLink/dataset/ 目录

常用的预训练数据集有：

Enwiki数据集: https://hf-mirror.com/datasets/lsb/enwiki20230101
C4数据集: https://hf-mirror.com/datasets/allenai/c4
ChineseWebText: https://hf-mirror.com/datasets/CASIA-LM/ChineseWebText

常用的对话指令微调数据集有：

单轮对话：Alpaca数据集: https://hf-mirror.com/datasets/tatsu-lab/alpaca
多轮对话：ShareGPT数据集: https://hf-mirror.com/datasets/shibing624/roleplay-zh-sharegpt-gpt4-data
多轮对话：AlpacaHistroy数据集: https://hf-mirror.com/datasets/lenML/oaast_rm_zh_jieba

数据集下载可以基于网页直接下载，也可以基于命令行下载，比如：

```bash
mkdir dataset
cd dataset/
wget https://huggingface.co/datasets/lsb/enwiki20230101/blob/main/data/train-00000-of-00042-d964455e17e96d5a.parquet
cd ..
```

### 3.2 数据集处理

#### 3.2.1 预训练数据集处理方法
```shell
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./dataset

python ./preprocess_data.py \
    --input ./dataset/train-00000-of-00042-d964455e17e96d5a.parquet \
    --tokenizer-name-or-path ./model_from_hf/llama2-hf \
    --tokenizer-type PretrainedFromHF \
    --handler-name GeneralPretrainHandler \
    --output-prefix ./dataset/enwiki \
    --json-keys text \
    --workers 4 \
    --log-interval 1000  
```

【--input】 可以直接输入到数据集目录或具体文件，如果是目录，则处理全部文件, 支持 .parquet \ .csv \ .json \ .jsonl \ .txt \ .arrow 格式， 同一个文件夹下的数据格式需要保持一致 

【--handler-name】 当前预训练默认使用 `GeneralPretrainHandler`，支持的是预训练数据风格，提取数据的`text`列，格式如下：

```shell
[
  {"text": "document"},
  {"other keys": "optional content"}
]
```

用户可结合具体数据处理需求添加新的Handler进行数据处理 

【--json-keys】 从文件中提取的列名列表，默认为 text，可以为 text, input, title 等多个输入，结合具体需求及数据集内容使用，如：

```bash
--json-keys text input output \
```

---

## 4. 大模型分布式预训练

### 4.1 配置预训练参数

预训练脚本保存在 example 中各模型文件夹下：pretrain_xxx_xx.sh
需根据实际情况修改路径和参数值

路径配置：包括**权重保存路径**、**权重加载路径**、**词表路径**、**数据集路径**
 ```shell
    # 根据实际情况配置词表、数据集、模型参数保存路径
    CKPT_SAVE_DIR="./ckpt"  #模型参数保存路径
    CKPT_LOAD_DIR="./model_weights/Grok1-mcore"  #权重加载路径
    TOKENIZER_MODEL="./model_from_hf/Grok1-hf/tokenizer.model"  #词表路径
    DATA_PATH="./dataset/enwiki_text_document"  #数据集路径
```
【--tokenizer-type】 

参数值为PretrainedFromHF时， 词表路径仅需要填到模型文件夹即可，不需要到tokenizer.model文件

【--data-path】 

支持多数据集训练，参数格式如下

```shell 
    --data-path dataset1-weight dataset1-path dataset2-weight dataset2-path
```

【单机运行】 
```shell
    GPUS_PER_NODE=8
    MASTER_ADDR=locahost
    MASTER_PORT=6000
    NNODES=1  
    NODE_RANK=0  
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```
【多机运行】 
```shell
    # 根据分布式集群实际情况配置分布式参数
    GPUS_PER_NODE=8  #每个节点的卡数
    MASTER_ADDR="your master node IP"  #都需要修改为主节点的IP地址（不能为localhost）
    MASTER_PORT=6000
    NNODES=2  #集群里的节点数，以实际情况填写,
    NODE_RANK="current node id"  #当前节点的RANK，多个节点不能重复，主节点为0, 其他节点可以是1,2..
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```
                      

### 4.2 启动预训练

```shell
    bash example/模型文件夹/pretrain_xxx_xxx.sh
```

**注意**：
- 多机训练需在多个终端同时启动预训练脚本(每个终端的预训练脚本只有NODE_RANK参数不同，其他参数均相同)
- 如果使用多机训练，且没有设置数据共享，需要在训练启动脚本中增加`--no-shared-storage`参数，设置此参数之后将会根据布式参数判断非主节点是否需要load数据，并检查相应缓存和生成数据


---

## 5. 大模型分布式推理

### 5.1 Generate：流式推理

ModelLink 流式推理脚本命名风格及启动方法为：
```shell
# 命名及启动：examples/model_name/generate_xxx.sh
bash examples/llama2/generate_llama2_7b_ptd.sh

```

```shell
# 按实际情况修改启动脚本中模型权重路径和分词器路径
CKPT_LOAD_DIR="./model_weights/Llama2-legacy/"
TOKENIZER_PATH="./model_from_hf/Llama2-hf/"

# 启动任务
bash examples/llama2/generate_llama2_7b_ptd.sh
```


---

## 6. 大模型分布式评估

ModelLink 基准评估脚本命名风格及启动方法为：
```shell
# 命名及启动：examples/model_name/evaluate_xxx.sh
bash examples/llama2/evaluate_llama2_7b_ptd.sh
```

```shell
# ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 修改模型参数路径和词表路径
TOKENIZER_PATH="./model_from_hf/llama2-hf/"  #词表路径
CHECKPOINT="./model_weights/llama2-legacy"  #模型路径
# 配置任务和数据集路径
DATA_PATH="./mmlu/data/test/"
TASK="mmlu"

# 启动评估脚本
bash examples/llama2/evaluate_llama2_7B_ptd.sh
```



