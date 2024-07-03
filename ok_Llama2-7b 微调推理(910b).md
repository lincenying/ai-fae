镜像: mindspore2.2.10-cann7.0.0beta1_py_3.9-euler_2.8.3_910b:v2
镜像源: swr.cn-east-292.mygaoxinai.com/huqs/mindspore2.2.10-cann7.0.0beta1_py_3.9-euler_2.8.3_910b:v2
Ascend: 4*ascend-d910b|CPU: 96核 768GB

# 1. 环境准备

## 1.1 安装 Mindformers

```bash
git clone -b dev https://gitee.com/mindspore/mindformers.git
cd mindformers
bash build.sh

cd /home/ma-user/work/
wget https://obs-community.obs.cn-north-1.myhuaweicloud.com/obsutil/current/obsutil_linux_arm64.tar.gz
tar -zxvf obsutil_linux_arm64.tar.gz
chmod +x ./obsutil_linux_arm64_5.5.12/obsutil
ln ./obsutil_linux_arm64_5.5.12/obsutil obsutil
/home/ma-user/work/obsutil config -i=###替换成AK### -k=###替换成SK### -e=obs.cn-east-292.mygaoxinai.com

```

# 2. 权重准备

## 2.1 创建权重存放目录及下载
```bash
mkdir -p /home/ma-user/work/mindformers/research/llama2/models
cd /home/ma-user/work/mindformers/research/llama2/models
# wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/llama2_7b.ckpt
/home/ma-user/work/obsutil cp obs://model-data/llama2/llama2_7b.ckpt ./
```
 
## 2.2 分词器文件下载
```bash
# wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/tokenizer.model
/home/ma-user/work/obsutil cp obs://model-data/llama2/tokenizer.model ./
```

# 3. 数据准备

## 3.2 下载原始数据

```bash
# wget https://github.com/tatsu-lab/stanford_alpaca/raw/main/alpaca_data.json
/home/ma-user/work/obsutil cp obs://model-data/llama2/alpaca_data.json ./
```

## 3.3 数据格式转换

命令行终端执行转换脚本:

```bash
python novelset2alpaca.py
```

继续执行 `alpaca_converter.py`，使用 `fastchat` 工具添加 `prompts` 模板，将原始数据集转换为 多轮对话格式:

```bash
python /home/ma-user/work/mindformers/mindformers/tools/dataset_preprocess/llama/alpaca_converter.py \
--data_path /home/ma-user/work/mindformers/research/llama2/models/alpaca_data.json \
--output_path /home/ma-user/work/mindformers/research/llama2/models/alpaca-data-conversation.json
```

## 3.4 分词器预处理并保存成 mindrecord 格式

执行 `llama_preprocess.py`，进行数据预处理、Mindrecord 数据生成，将带有 prompt 模板的
数据转换为 mindrecord 格式

```bash
cd /home/ma-user/work/mindformers/
# 这步可能时间有点长
python /home/ma-user/work/mindformers/mindformers/tools/dataset_preprocess/llama/llama_preprocess.py \
--dataset_type qa \
--input_glob /home/ma-user/work/mindformers/research/llama2/models/alpaca-data-conversation.json \
--model_file /home/ma-user/work/mindformers/research/llama2/models/tokenizer.model \
--seq_length 2048 \
--output_file /home/ma-user/work/mindformers/research/llama2/models/alpaca-fastchat2048.mindrecord

```

# 4. 模型训练

## 4.1 修改配置文件

```bash
cp /home/ma-user/work/mindformers/configs/llama2/run_llama2_7b_lora_910b.yaml /home/ma-user/work/mindformers/research/llama2/run_llama2_7b_lora_910b.yaml
```

### 4.1.1 配置权重路径
将
```yaml
load_checkpoint: '/home/ma-user/work/mindformers/research/llama2/models/llama2_7b.ckpt'
use_parallel: False

train_dataset: &train_dataset
  data_loader:
    dataset_dir: "/home/ma-user/work/mindformers/research/llama2/models/alpaca-fastchat2048.mindrecord"

callbacks:
  - type: MFLossMonitor
  - type: CheckpointMointor
    # 10000次执行后保存一次
    save_checkpoint_steps: 10000
    # 保存3次
    integrated_save: 1
```


## 4.2 启动训练脚本

```bash
cd /home/ma-user/work/mindformers/scripts
# 单卡启动
bash run_standalone.sh /home/ma-user/work/mindformers/research/llama2/run_llama2_7b_lora_910b.yaml 0 finetune
# 多卡启动
bash run_distribute.sh /user/config/jobstart_hccl.json /home/ma-user/work/mindformers/research/llama2/run_llama2_7b_lora_910b.yaml [0,8] finetune
```

训练进程后台运行，可在`/home/ma-user/work/mindformers/output/log/rank_0` 目录下查看实时更新的训练日志

# 5. 模型推理

## 5.1 修改配置文件

```bash
cp /home/ma-user/work/mindformers/configs/llama2/run_llama2_7b_910b.yaml /home/ma-user/work/mindformers/research/llama2/run_llama2_7b_910b.yaml

vi /home/ma-user/work/mindformers/research/llama2/run_llama2_7b_910b.yaml
```

修改权重初始化类型，将默认的 ”float16” 改为 ”float32”，保障推理精度(非必须，精度要求高场 景下可开启)

```yaml
param_init_type: "float16"
```
改成
```yaml
param_init_type: "float32"
```

```yaml
load_checkpoint: '/home/ma-user/work/mindformers/research/llama2/models/llama2_7b.ckpt'
run_mode: 'predict'
train_dataset: &train_dataset
  data_loader:
    dataset_dir: "/home/ma-user/work/mindformers/research/llama2/models/alpaca-fastchat2048.mindrecord"
model:
  model_config:
    seq_length: 2048
processor:
  tokenizer:
    vocab_file: "/home/ma-user/work/mindformers/research/llama2/models/tokenizer.model"
```

## 5.2 执行推理脚本

```bash
cd /home/ma-user/work/mindformers

python run_mindformer.py --config /home/ma-user/work/mindformers/research/llama2/run_llama2_7b_910b.yaml --run_mode predict --predict_data 'I love Beijing, because' --use_parallel False
```