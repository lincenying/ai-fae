- 镜像: mindspore2.2.10-cann7.0.0beta1_py_3.9-euler_2.8.3_910b:v2
- 镜像源: swr.cn-east-292.mygaoxinai.com/huqs/mindspore2.2.10-cann7.0.0beta1_py_3.9-euler_2.8.3_910b:v2
- Ascend: 4*ascend-d910b|CPU: 96核 768GB


# 1. 环境准备

## 1.1 安装 Mindformers

```bash
git clone -b r1.0 https://gitee.com/mindspore/mindformers.git
cd mindformers
bash build.sh

```

## 1.2 安装依赖

```bash
pip install torch transformers transformers_stream_generator einops accelerate tiktoken

```

## 1.3 安装obsutil

```bash
cd /home/ma-user/work/
wget https://obs-community.obs.cn-north-1.myhuaweicloud.com/obsutil/current/obsutil_linux_arm64.tar.gz
tar -zxvf obsutil_linux_arm64.tar.gz
chmod +x ./obsutil_linux_arm64_5.5.12/obsutil
ln ./obsutil_linux_arm64_5.5.12/obsutil obsutil
export OBSAK="这里改成AK"
export OBSSK="这里改成SK"
/home/ma-user/work/obsutil config -i=${OBSAK} -k=${OBSSK} -e=obs.cn-east-292.mygaoxinai.com

```


# 2. 权重准备

## 2.1 创建权重存放目录及下载
```bash
mkdir -p /home/ma-user/work/mindformers/research/internlm/models
cd /home/ma-user/work/mindformers/research/internlm/models
# wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/internlm/internlm-chat.ckpt
/home/ma-user/work/obsutil cp obs://model-data/internlm/internlm-chat.ckpt ./

```
 
## 2.2 分词器文件下载
```bash
# wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/internlm/tokenizer.model
/home/ma-user/work/obsutil cp obs://model-data/internlm/tokenizer.model ./
```

# 3. 数据准备

## 3.2 下载原始数据

```bash
cd /home/ma-user/work/mindformers/research/internlm/models
/home/ma-user/work/obsutil cp obs://model-data/internlm/alpaca_gpt4_data_zh.json ./
```

## 3.3 数据格式转换

wiki_data_preprocess.py，将原始数据集转换为指定格式。

```bash
cd /home/ma-user/work/mindformers/research/internlm

python alpaca_data_preprocess.py \
--mindrecord_schema internlm_alpaca \
--input_glob /home/ma-user/work/mindformers/research/internlm/models/alpaca_gpt4_data_zh.json \
--output_file /home/ma-user/work/mindformers/research/internlm/models/alpaca.mindrecord \
--model_file /home/ma-user/work/mindformers/research/internlm/models/tokenizer.model \
--seq_length 2048
```

# 4. 模型训练

## 4.1 修改配置文件

```bash
mkdir -p  /home/ma-user/work/mindformers/research/internlm/models/rank_0
mv /home/ma-user/work/mindformers/research/internlm/models/internlm-chat.ckpt /home/ma-user/work/mindformers/research/internlm/models/rank_0/internlm-chat.ckpt
vi /home/ma-user/work/mindformers/research/internlm/run_internlm_7b_lora_910b.yaml
```

### 4.1.1 配置权重路径

```yaml
load_checkpoint: '/home/ma-user/work/mindformers/research/internlm/models/'

train_dataset: &train_dataset
  data_loader:
    dataset_dir: "/home/ma-user/work/mindformers/research/internlm/models/alpaca.mindrecord"

model:
  model_config:
    seq_length: 2048

processor:
  tokenizer:
    vocab_file: "/home/ma-user/work/mindformers/research/internlm/models/tokenizer.model"
```

## 4.2 启动训练脚本

```bash
cd /home/ma-user/work/mindformers/research

bash run_singlenode.sh \
"python internlm/run_internlm.py \
--config internlm/run_internlm_7b_lora_910b.yaml \
--run_mode finetune \
--use_parallel True \
--load_checkpoint /home/ma-user/work/mindformers/research/internlm/models/ \
--auto_trans_ckpt True \
--train_dataset /home/ma-user/work/mindformers/research/internlm/models/alpaca.mindrecord" \
/user/config/jobstart_hccl.json [0,4] 4
```

训练进程后台运行，可在`/home/ma-user/work/mindformers/research/output/log/rank_0` 目录下查看实时更新的训练日志

# 5. 模型推理

## 5.1 单卡推理

```bash
vi /home/ma-user/work/mindformers/research/internlm/run_internlm_7b_910b.yaml
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
# 填写权重路径
load_checkpoint: '/home/ma-user/work/mindformers/research/internlm/models/rank_0/internlm-chat.ckpt'
# 关闭自动权重转换
auto_trans_ckpt: False
# 关闭并行模式
use_parallel: False
# 使用增量推理
model:
  model_config:
    use_past: True
# 配置词表路径
processor:
  tokenizer:
    vocab_file: "/home/ma-user/work/mindformers/research/internlm/models/tokenizer.model"

```

```bash
cd /home/ma-user/work/mindformers/research/internlm/
export PYTHONPATH=/home/ma-user/work/mindformers:$PYTHONPATH

python run_internlm.py \
--config "run_internlm_7b_910b.yaml" \
--run_mode predict \
--use_parallel False \
--load_checkpoint /home/ma-user/work/mindformers/research/internlm/models/rank_0/internlm-chat.ckpt \
--predict_data '我们来对对联吧！生意如春意 的下联是' \
--device_id 0

```
