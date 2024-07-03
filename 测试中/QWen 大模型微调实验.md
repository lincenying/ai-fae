# 1. 环境准备

## 1.1 安装 Mindformers

```bash
git clone -b dev https://gitee.com/mindspore/mindformers.git
cd mindformers
bash build.sh
```

# 2. 权重准备

## 2.1 创建权重存放目录及下载
```bash
mkdir -p /home/ma-user/work/mindformers/models/qwen
cd models/qwen
wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/qwen/qwen_7b_base.ckpt
 ```
 
## 2.2 分词器文件下载
```bash
wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/qwen/qwen.tiktoken
```

# 3. 数据准备

## 3.1 创建数据存放目录
 
 ```bash
mkdir -p /home/ma-user/work/mindformers/datasets/novelset
cd /home/ma-user/work/mindformers/datasets/novelset
```

## 3.2 下载原始数据

```bash
wget https://github.com/tatsu-lab/stanford_alpaca/raw/main/alpaca_data.json
```

## 3.3 数据格式转换

执行alpaca_converter.py，将原始数据集转换为指定格式。

```bash
cd /home/ma-user/work/mindformers/

python research/qwen/alpaca_converter.py \
--data_path /home/ma-user/work/mindformers/datasets/novelset/alpaca_data.json \
--output_path /home/ma-user/work/mindformers/datasets/novelset/alpaca-data-conversation.json
```

执行qwen_preprocess.py，进行数据预处理和Mindrecord数据生成

```bash
cd /home/ma-user/work/mindformers/

python research/qwen/qwen_preprocess.py \
--input_glob /home/ma-user/work/mindformers/datasets/novelset/alpaca-data-conversation.json \
--model_file /home/ma-user/work/mindformers/models/qwen/qwen.tiktoken \
--seq_length 2048 \
--output_file /home/ma-user/work/mindformers/models/qwen/alpaca.mindrecord
```

# 4. 模型训练

## 4.1 修改配置文件

```bash
vi /home/ma-user/work/mindformers/configs/llama2/run_llama2_7b_lora_910b.yaml
```

### 4.1.1 配置权重路径
将
```yaml
load_checkpoint: ''
```
改成
```yaml
load_checkpoint: '/home/ma-user/work/mindformers/models/llama2/llama2_7b.ckpt'
```

### 4.1.2 配置训练数据路径
将
```yaml
dataset_dir: ""
```
改成
```yaml
dataset_dir: "/home/ma-user/work/mindformers/datasets/novelset/novelset.mindrecord"
```

### 4.1.3 关闭分布式并行
将
```yaml
use_parallel: True
```
改成
```yaml
use_parallel: False
```

## 4.2 启动训练脚本

```bash
cd /home/ma-user/work/mindformers/scripts
bash run_standalone.sh ../configs/llama2/run_llama2_7b_lora_910b.yaml 0 finetune
```

训练进程后台运行，可在`/home/ma-user/work/mindformers/output/log/rank_0` 目录下查看实时更新的训练日志

# 5. 模型推理

## 5.1 修改配置文件

```bash
vi /home/ma-user/work/mindformers/configs/llama2/run_llama2_7b_lora_910b.yaml
```

修改权重初始化类型，将默认的 ”float16” 改为 ”float32”，保障推理精度(非必须，精度要求高场 景下可开启)

```yaml
param_init_type: "float16"
```
改成
```yaml
param_init_type: "float32"
```

## 5.2 新建推理脚本

在`work`目录下新建推理脚本: `llama2_7b_infer.py`

```bash
cd /home/ma-user/work/
vi llama2_7b_infer.py
```

```py
import argparse
import mindspore as ms
import numpy as np
import os
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train import Model
from mindformers import MindFormerConfig, LlamaConfig, TransformerOpParallelConfig, AutoTokenizer, LlamaForCausalLM, AutoProcessor
from mindformers import init_context, ContextConfig, ParallelContextConfig
from mindformers.tools.utils import str2bool, get_real_rank
from mindformers.trainer.utils import get_last_checkpoint
from mindformers.pet import get_pet_model, LoraConfig

# 多batch输入
prompt_input_temple = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
)
prompt_no_input_temple = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)
inputs = [
    prompt_input_temple.format_map({"instruction": "请将给你的文本内容扩写成新闻", "input": "杭州某公司员工中1亿元彩票大奖。"}),
    prompt_no_input_temple.format_map({"instruction": "请帮我制定一份详细的健身计划?"}),
]
# set model config
llama2_7b_config_path = "/home/ma-user/work/mindformers/configs/llama2/run_llama2_7b_lora_910b.yaml"
config = MindFormerConfig(llama2_7b_config_path)
# 初始化环境
init_context(use_parallel=config.use_parallel, context_config=config.context, parallel_config=config.parallel)
model_config = LlamaConfig(**config.model.model_config)
model_config.parallel_config = TransformerOpParallelConfig(**config.parallel_config)
model_config.batch_size = 1  # len(inputs)
model_config.use_past = True
model_config.seq_length = 512  # 推理时可适当改小，提升推理速度
model_config.do_sample = True  # 开启后，减少推理结果中词语重复的情况
model_config.checkpoint_name_or_path = "/home/ma-user/work/mindformers/models/llama2/llama2_7b.ckpt"
print(f"config is: {model_config}")
# build tokenizer
# 在线加载
tokenizer = AutoTokenizer.from_pretrained("llama2_7b")
# build model from config
model = LlamaForCausalLM(model_config)
# lora 权重初始化
if model_config.pet_config:
    print("----------------Init lora params----------------")
    pet_config = LoraConfig(
        lora_rank=config.model.model_config.pet_config.lora_rank,
        lora_alpha=config.model.model_config.pet_config.lora_alpha,
        lora_dropout=config.model.model_config.pet_config.lora_dropout,
        target_modules=config.model.model_config.pet_config.target_modules,
    )
    model = get_pet_model(model, pet_config)
# inputs_ids = tokenizer(inputs, max_length=model_config.seq_length, padding="max_length")["input_ids"] # outputs = model.generate(inputs_ids,
# max_length=model_config.max_decode_length,
# do_sample=model_config.do_sample,
# top_k=model_config.top_k,
# top_p=model_config.top_p)
# for output in outputs:
#   print(tokenizer.decode(output))
for example_input in inputs:
    input_ids = tokenizer([example_input], max_length=model_config.seq_length, padding="max_length")["input_ids"]
    outputs = model.generate(
        input_ids, max_length=model_config.max_decode_length, do_sample=model_config.do_sample, top_k=model_config.top_k, top_p=model_config.top_p
    )
    print(tokenizer.decode(outputs[0]))

```

## 5.3 执行推理脚本

```bash
cd /home/ma-user/work
python ./llama2_7b_infer.py
```