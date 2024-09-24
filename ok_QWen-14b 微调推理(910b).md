[当前文档访问路径](https://ai-fae.readthedocs.io/zh-cn/latest/ok_QWen-14b%20微调推理(910b).html)

- 镜像: mindspore2.2.14-cann7.0.0beta1_py_3.9-euler_2.8.3_910:v3
- 镜像源: swr.cn-east-292.mygaoxinai.com/huqs/mindspore2.2.14-cann7.0.0beta1_py_3.9-euler_2.8.3_910:v3
- 规格: 4*ascend-d910b|CPU: 96核 768GB


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
mkdir -p /home/ma-user/work/mindformers/research/qwen/14b_base/
cd /home/ma-user/work/mindformers/research/qwen/14b_base/
```

2.1.1 / 2.1.2 根据情况 2选1 即可

### 2.1.1 直接使用转换完成的权重
```bash
# wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/qwen/qwen_14b_base.ckpt
obsutil cp obs://model-data/qianwen/qwen_14b_base.ckpt ./
mkdir -p  /home/ma-user/work/mindformers/research/qwen/14b_base/rank_0
mv /home/ma-user/work/mindformers/research/qwen/14b_base/qwen_14b_base.ckpt /home/ma-user/work/mindformers/research/qwen/14b_base/rank_0/qwen_14b_base.ckpt

```

### 2.1.2 使用huggingface权重自行转换

#### 2.1.2.1 使用huggingface下载权重

```bash
# 1. 使用魔搭下载
pip install modelscope

cd /home/ma-user/work/mindformers/research/qwen/
modelscope download --model 'qwen/Qwen-14B' --local_dir './14b_base'


# 2. 通过hf-mirror克隆文件
git clone https://hf-mirror.com/Qwen/Qwen-14B
# 删除无效的权重
rm -rf ./Qwen-14B/*.safetensors
mv Qwen-14B 14b_base
cd 14b_base

# 2.1 / 2.2 二选一下载即可

# 2.1 使用hf-mirror下载权重
wget -O model-00001-of-00015.safetensors https://hf-mirror.com/Qwen/Qwen-14B/resolve/main/model-00001-of-00015.safetensors?download=true
wget -O model-00002-of-00015.safetensors https://hf-mirror.com/Qwen/Qwen-14B/resolve/main/model-00002-of-00015.safetensors?download=true
wget -O model-00003-of-00015.safetensors https://hf-mirror.com/Qwen/Qwen-14B/resolve/main/model-00003-of-00015.safetensors?download=true
wget -O model-00004-of-00015.safetensors https://hf-mirror.com/Qwen/Qwen-14B/resolve/main/model-00004-of-00015.safetensors?download=true
wget -O model-00005-of-00015.safetensors https://hf-mirror.com/Qwen/Qwen-14B/resolve/main/model-00005-of-00015.safetensors?download=true
wget -O model-00006-of-00015.safetensors https://hf-mirror.com/Qwen/Qwen-14B/resolve/main/model-00006-of-00015.safetensors?download=true
wget -O model-00007-of-00015.safetensors https://hf-mirror.com/Qwen/Qwen-14B/resolve/main/model-00007-of-00015.safetensors?download=true
wget -O model-00008-of-00015.safetensors https://hf-mirror.com/Qwen/Qwen-14B/resolve/main/model-00008-of-00015.safetensors?download=true
wget -O model-00008-of-00015.safetensors https://hf-mirror.com/Qwen/Qwen-14B/resolve/main/model-00009-of-00015.safetensors?download=true
wget -O model-00008-of-00015.safetensors https://hf-mirror.com/Qwen/Qwen-14B/resolve/main/model-00010-of-00015.safetensors?download=true
wget -O model-00008-of-00015.safetensors https://hf-mirror.com/Qwen/Qwen-14B/resolve/main/model-00011-of-00015.safetensors?download=true
wget -O model-00008-of-00015.safetensors https://hf-mirror.com/Qwen/Qwen-14B/resolve/main/model-00012-of-00015.safetensors?download=true
wget -O model-00008-of-00015.safetensors https://hf-mirror.com/Qwen/Qwen-14B/resolve/main/model-00013-of-00015.safetensors?download=true
wget -O model-00008-of-00015.safetensors https://hf-mirror.com/Qwen/Qwen-14B/resolve/main/model-00014-of-00015.safetensors?download=true
wget -O model-00008-of-00015.safetensors https://hf-mirror.com/Qwen/Qwen-14B/resolve/main/model-00015-of-00015.safetensors?download=true

# 2.2 通过obs下载权重

obsutil cp obs://model-data/qianwen/14b_base/model-00001-of-00015.safetensors ./
obsutil cp obs://model-data/qianwen/14b_base/model-00002-of-00015.safetensors ./
obsutil cp obs://model-data/qianwen/14b_base/model-00003-of-00015.safetensors ./
obsutil cp obs://model-data/qianwen/14b_base/model-00004-of-00015.safetensors ./
obsutil cp obs://model-data/qianwen/14b_base/model-00005-of-00015.safetensors ./
obsutil cp obs://model-data/qianwen/14b_base/model-00006-of-00015.safetensors ./
obsutil cp obs://model-data/qianwen/14b_base/model-00007-of-00015.safetensors ./
obsutil cp obs://model-data/qianwen/14b_base/model-00008-of-00015.safetensors ./
obsutil cp obs://model-data/qianwen/14b_base/model-00009-of-00015.safetensors ./
obsutil cp obs://model-data/qianwen/14b_base/model-00010-of-00015.safetensors ./
obsutil cp obs://model-data/qianwen/14b_base/model-00011-of-00015.safetensors ./
obsutil cp obs://model-data/qianwen/14b_base/model-00012-of-00015.safetensors ./
obsutil cp obs://model-data/qianwen/14b_base/model-00013-of-00015.safetensors ./
obsutil cp obs://model-data/qianwen/14b_base/model-00014-of-00015.safetensors ./
obsutil cp obs://model-data/qianwen/14b_base/model-00015-of-00015.safetensors ./

```

#### 2.1.2.2 torch权重转mindspore权重

转换权重
```bash
cd /home/ma-user/work/mindformers/

python research/qwen/convert_weight.py \
--torch_ckpt_dir ./research/qwen/14b_base/ \
--mindspore_ckpt_path ./research/qwen/14b_base/qwen_14b_base.ckpt
```

如果报错:
ImportError: /home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch/lib/../../torch.libs/libgomp-d22c30c5.so.1.0.0: cannot allocate memory in static TLS block
执行:
```bash
export LD_PRELOAD='/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch.libs/libgomp-4dbbc2f2.so.1.0.0'
```

```bash
mkdir -p  /home/ma-user/work/mindformers/research/qwen/14b_base/rank_0
mv /home/ma-user/work/mindformers/research/qwen/14b_base/qwen_14b_base.ckpt /home/ma-user/work/mindformers/research/qwen/14b_base/rank_0/qwen_14b_base.ckpt
```
 
## 2.2 分词器文件下载
```bash
cd /home/ma-user/work/mindformers/research/qwen/14b
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
--model_file /home/ma-user/work/mindformers/research/qwen/14b_base/qwen.tiktoken \
--seq_length 2048 \
--output_file /home/ma-user/work/mindformers/research/qwen/data/alpaca.mindrecord

```

# 4. 模型训练

## 4.1 修改配置文件

```bash

vi /home/ma-user/work/mindformers/research/qwen/run_qwen_14b_lora.yaml
```

### 4.1.1 配置权重路径

```yaml
load_checkpoint: '/home/ma-user/work/mindformers/research/qwen/14b_base/'

train_dataset: &train_dataset
  data_loader:
    dataset_dir: "/home/ma-user/work/mindformers/research/qwen/data/alpaca.mindrecord"

model:
  model_config:
    seq_length: 2048

processor:
  tokenizer:
    vocab_file: "/home/ma-user/work/mindformers/research/qwen/14b_base/qwen.tiktoken"
```

## 4.2 启动训练脚本

```bash
cd /home/ma-user/work/mindformers/research/qwen/14b_base/

export MS_ASCEND_CHECK_OVERFLOW_MODE=INFNAN_MODE

bash /home/ma-user/work/mindformers/research/run_singlenode.sh \
"python /home/ma-user/work/mindformers/research/qwen/run_qwen.py \
--config /home/ma-user/work/mindformers/research/qwen/run_qwen_14b_lora.yaml \
--load_checkpoint /home/ma-user/work/mindformers/research/qwen/14b_base/ \
--use_parallel True \
--run_mode finetune \
--auto_trans_ckpt True \
--train_data /home/ma-user/work/mindformers/research/qwen/data/alpaca.mindrecord" \
/user/config/jobstart_hccl.json [0,8] 8

```

训练进程后台运行，可在`/home/ma-user/work/mindformers/output/log/rank_0` 目录下查看实时更新的训练日志

# 5. 模型推理

## 5.1 单卡推理

```bash
# 注意, 如果权重用lora微调过, 这里的配置需要用lora微调的配置文件做修改
# vi /home/ma-user/work/mindformers/research/qwen/run_qwen_14b_lora.yaml
vi /home/ma-user/work/mindformers/research/qwen/run_qwen_14b.yaml
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
load_checkpoint: '/home/ma-user/work/mindformers/research/qwen/14b_base/'
# 关闭自动权重转换
auto_trans_ckpt: False
# 关闭并行模式
use_parallel: False
# 使用增量推理
model:
  model_config:
    checkpoint_name_or_path: "/home/ma-user/work/mindformers/research/qwen/14b_base/rank_0/qwen_14b_base.ckpt"
    use_past: True
# 配置词表路径
processor:
  tokenizer:
    vocab_file: "/home/ma-user/work/mindformers/research/qwen/14b_base/qwen.tiktoken"

```

```bash
export PYTHONPATH=/home/ma-user/work/mindformers:$PYTHONPATH

# Atlas 800T A2上运行时需要设置如下环境变量，否则推理结果会出现精度问题
export MS_GE_TRAIN=0
export MS_ENABLE_GE=1
export MS_ENABLE_REF_MODE=1

cd /home/ma-user/work/mindformers/research/qwen/14b_base

# 注意, 如果权重用lora微调过, 这里的配置需要用lora微调的配置文件
# --config /home/ma-user/work/mindformers/research/qwen/run_qwen_14b_lora.yaml
#单卡推理
python /home/ma-user/work/mindformers/research/qwen/run_qwen.py \
--config /home/ma-user/work/mindformers/research/qwen/run_qwen_14b.yaml \
--predict_data '小孩挑食怎么办?' \
--run_mode predict \
--auto_trans_ckpt False \
--use_parallel False \
--load_checkpoint /home/ma-user/work/mindformers/research/qwen/14b-chat/rank_0/qwen_14b_chat.ckpt \
--device_id 0

# 多卡推理
bash /home/ma-user/work/mindformers/research/run_singlenode.sh \
"python /home/ma-user/work/mindformers/research/qwen/run_qwen.py \
--config /home/ma-user/work/mindformers/research/qwen/run_qwen_14b.yaml \
--run_mode predict \
--use_parallel True \
--load_checkpoint /home/ma-user/work/mindformers/research/qwen/14b_base/rank_0/qwen_14b_base.ckpt \
--auto_trans_ckpt True \
--predict_data  APG除盐床隔离排空后投运的操作过程" \
/user/config/jobstart_hccl.json [0,2] 2
```

# 6. C-Eval 评测

## 6.1 下载eval py文件

```bash
cd /home/ma-user/work/mindformers/research/qwen/
obsutil cp obs://model-data/eval.zip ./
unzip eval.zip
```

## 6.2 修改配置文件

文件`eval_utils.py`:

```py
group.add_argument('--config', default='run_qwen_7b.yaml', type=str, help='Config file path. (default: ./run_qwen_7b.yaml)')
```
修改`config`参数为对应的推理配置文件, 如:`run_qwen_14b.yaml`
```py
group.add_argument('--config', default='run_qwen_14b.yaml', type=str, help='Config file path. (default: ./run_qwen_7b.yaml)')
```

## 6.3 运行评测脚本
```bash
cd /home/ma-user/work/mindformers/research/qwen/
obsutil cp obs://model-data/ceval-exam.zip ./
mkdir -p data/ceval && cd data/ceval
unzip ../../ceval-exam.zip && cd ../../
python eval/evaluate_ceval.py -d data/ceval/

```

# [过滤权重和权重合并](https://ai-fae.readthedocs.io/zh-cn/latest/过滤权重和权重合并.html)