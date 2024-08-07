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
# 下载obsutil
wget https://obs-community.obs.cn-north-1.myhuaweicloud.com/obsutil/current/obsutil_linux_arm64.tar.gz
# 解压缩obsutil
tar -zxvf obsutil_linux_arm64.tar.gz
# 修改可执行文件
chmod +x ./obsutil_linux_arm64_5.5.12/obsutil
# 移动obsutil
mv ./obsutil_linux_arm64_5.5.12 ./obs_bin
# 添加环境变量
export OBSAK="这里改成AK"
export OBSSK="这里改成SK"
# notebook停止后也需要重新执行下面两条命令
export PATH=$PATH:/home/ma-user/work/obs_bin
obsutil config -i=${OBSAK} -k=${OBSSK} -e=obs.cn-east-292.mygaoxinai.com

```


# 2. 权重准备

## 2.1 创建权重存放目录及下载
```bash
mkdir -p /home/ma-user/work/mindformers/research/qwen/7b/
cd /home/ma-user/work/mindformers/research/qwen/7b/
```

### 2.1.1 直接使用转换完成的权重
```bash
# wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/qwen/qwen_7b_base.ckpt
obsutil cp obs://model-data/qianwen/qwen_7b_base.ckpt ./

```

### 2.1.2 使用huggingface权重自行转换

```bash
# 1. 使用魔塔下载
pip install modelscope

cd /home/ma-user/work/mindformers/research/qwen/
vi down.py
#--->写入以下内容
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen-7B')
#<---保存文件
python down.py

mv /home/ma-user/.cache/modelscope/hub/qwen/Qwen-7B ./7b


# 2. 通过hf-mirror克隆文件
git clone https://hf-mirror.com/Qwen/Qwen-7B
# 删除无效的权重
rm -rf ./Qwen-7B/*.safetensors
mv Qwen-7B 7b
cd 7b

# 2.1 使用hf-mirror下载权重
wget -O model-00001-of-00008.safetensors https://hf-mirror.com/Qwen/Qwen-7B/resolve/main/model-00001-of-00004.safetensors?download=true
wget -O model-00002-of-00004.safetensors https://hf-mirror.com/Qwen/Qwen-7B/resolve/main/model-00002-of-00004.safetensors?download=true
wget -O model-00003-of-00004.safetensors https://hf-mirror.com/Qwen/Qwen-7B/resolve/main/model-00003-of-00004.safetensors?download=true
wget -O model-00004-of-00004.safetensors https://hf-mirror.com/Qwen/Qwen-7B/resolve/main/model-00004-of-00004.safetensors?download=true

# 2.2 通过obs下载权重

mkdir -p /home/ma-user/work/mindformers/research/qwen/7b
cd /home/ma-user/work/mindformers/research/qwen/7b
obsutil cp obs://model-data/qianwen1.5/7b/base/model-00001-of-00004.safetensors ./
obsutil cp obs://model-data/qianwen1.5/7b/base/model-00002-of-00004.safetensors ./
obsutil cp obs://model-data/qianwen1.5/7b/base/model-00003-of-00004.safetensors ./
obsutil cp obs://model-data/qianwen1.5/7b/base/model-00004-of-00004.safetensors ./

```

## 2.2 torch权重转mindspore权重

转换权重
```bash
cd /home/ma-user/work/mindformers/

python research/qwen/convert_weight.py \
--torch_ckpt_dir ./research/qwen/14b/ \
--mindspore_ckpt_path ./research/qwen/14b/qwen_14b_base.ckpt
```

如果报错:
ImportError: /home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch/lib/../../torch.libs/libgomp-d22c30c5.so.1.0.0: cannot allocate memory in static TLS block
执行:
```bash
export LD_PRELOAD='/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch.libs/libgomp-d22c30c5.so.1.0.0'
```
 
## 2.3 分词器文件下载
```bash
# wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/qwen/qwen.tiktoken
obsutil cp obs://model-data/qianwen/qwen.tiktoken ./
```

# 3. 数据准备

## 3.2 下载原始数据

```bash
mkdir /home/ma-user/work/mindformers/research/qwen/data
cd /home/ma-user/work/mindformers/research/qwen/data
# wget https://github.com/tatsu-lab/stanford_alpaca/raw/main/alpaca_data.json
obsutil cp obs://model-data/qianwen/alpaca_data.json ./
```

## 3.3 数据格式转换

执行alpaca_converter.py，将原始数据集转换为指定格式。

```bash
cd /home/ma-user/work/mindformers/

python research/qwen/alpaca_converter.py \
--data_path /home/ma-user/work/mindformers/research/qwen/data/alpaca_data.json \
--output_path /home/ma-user/work/mindformers/research/qwen/data/alpaca-data-conversation.json

```

执行qwen_preprocess.py，进行数据预处理和Mindrecord数据生成

```bash
cd /home/ma-user/work/mindformers/

python research/qwen/qwen_preprocess.py \
--input_glob /home/ma-user/work/mindformers/research/qwen/data/alpaca-data-conversation.json \
--model_file /home/ma-user/work/mindformers/research/qwen/7b/qwen.tiktoken \
--seq_length 2048 \
--output_file /home/ma-user/work/mindformers/research/qwen/data/alpaca.mindrecord

```

# 4. 模型训练

## 4.1 修改配置文件

```bash
mkdir -p  /home/ma-user/work/mindformers/research/qwen/7b/rank_0
mv /home/ma-user/work/mindformers/research/qwen/7b/qwen_7b_base.ckpt /home/ma-user/work/mindformers/research/qwen/7b/rank_0/qwen_7b_base.ckpt
vi /home/ma-user/work/mindformers/research/qwen/run_qwen_7b_lora.yaml

```

### 4.1.1 配置权重路径

```yaml
load_checkpoint: '/home/ma-user/work/mindformers/research/qwen/7b/'

train_dataset: &train_dataset
  data_loader:
    dataset_dir: "/home/ma-user/work/mindformers/research/qwen/data/alpaca.mindrecord"

model:
  model_config:
    seq_length: 2048

processor:
  tokenizer:
    vocab_file: "/home/ma-user/work/mindformers/research/qwen/7b/qwen.tiktoken"
```

## 4.2 启动训练脚本

```bash
cd /home/ma-user/work/mindformers/research/qwen/7b/

export MS_ASCEND_CHECK_OVERFLOW_MODE=INFNAN_MODE

bash /home/ma-user/work/mindformers/research/run_singlenode.sh
"python /home/ma-user/work/mindformers/research/qwen/run_qwen.py \
--config /home/ma-user/work/mindformers/research/qwen/run_qwen_7b_lora.yaml \
--load_checkpoint /home/ma-user/work/mindformers/research/qwen/7b/ \
--use_parallel True \
--run_mode finetune \
--auto_trans_ckpt True \
--train_data /home/ma-user/work/mindformers/research/qwen/data/alpaca.mindrecord" \
/user/config/jobstart_hccl.json [0,4] 4

```

训练进程后台运行，可在`/home/ma-user/work/mindformers/output/log/rank_0` 目录下查看实时更新的训练日志

# 5. 模型推理

## 5.1 单卡推理

```bash
vi /home/ma-user/work/mindformers/research/qwen/run_qwen_7b.yaml
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
load_checkpoint: '/home/ma-user/work/mindformers/research/qwen/7b/'
# 关闭自动权重转换
auto_trans_ckpt: False
# 关闭并行模式
use_parallel: False
# 使用增量推理
model:
  model_config:
    checkpoint_name_or_path: '/home/ma-user/work/mindformers/research/qwen/7b/rank_0/qwen_7b_base.ckpt'
    use_past: True
# 配置词表路径
processor:
  tokenizer:
    vocab_file: "/home/ma-user/work/mindformers/research/qwen/7b/qwen.tiktoken"

```

```bash
export PYTHONPATH=/home/ma-user/work/mindformers:$PYTHONPATH

cd /home/ma-user/work/mindformers/research/qwen/7b

# 单卡推理
python /home/ma-user/work/mindformers/research/qwen/run_qwen.py \
--config /home/ma-user/work/mindformers/research/qwen/run_qwen_7b.yaml \
--predict_data '如何治疗口腔溃疡' \
--run_mode predict \
--load_checkpoint /home/ma-user/work/mindformers/research/qwen/7b/rank_0/qwen_7b_base.ckpt \
--device_id 0

# 多卡推理

bash /home/ma-user/work/mindformers/research/run_singlenode.sh \
"python /home/ma-user/work/mindformers/research/qwen/run_qwen.py \
--config /home/ma-user/work/mindformers/research/qwen/run_qwen_7b.yaml \
--run_mode predict \
--use_parallel True \
--load_checkpoint /home/ma-user/work/mindformers/research/qwen/7b/rank_0/qwen_7b_base.ckpt \
--auto_trans_ckpt True \
--predict_data 比较适合深度学习入门的书籍有" \
/user/config/jobstart_hccl.json [0,2] 2
```
