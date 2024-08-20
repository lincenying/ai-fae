镜像: mindspore2.2.14-cann7.0.0beta1_py_3.9-euler_2.8.3_910:v2_qwen1_5_72b
Ascend: 8*ascend-d910b|CPU: 192核 1536GB



## 1.1 安装 Mindformers

```bash
git clone -b r1.0 https://gitee.com/mindspore/mindformers.git # 建议选这个版本
cd mindformers
bash build.sh

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
cd /home/ma-user/work/mindformers/research/qwen1_5

################
# 下面方法3选1即可#
################

# 1. 使用魔搭下载
pip install modelscope

modelscope download --model 'qwen/Qwen1.5-14B' --local_dir './14b'

# 2. 通过hf-mirror克隆文件
git clone https://hf-mirror.com/Qwen/Qwen1.5-14B
# 删除无效的权重
rm -rf ./Qwen1.5-14B-Chat/*.safetensors
mv Qwen1.5-14B-Chat 14b_base
cd 14b_base

wget -O model-00001-of-00008.safetensors https://hf-mirror.com/Qwen/Qwen1.5-14B-Chat/resolve/main/model-00001-of-00008.safetensors?download=true
wget -O model-00002-of-00008.safetensors https://hf-mirror.com/Qwen/Qwen1.5-14B-Chat/resolve/main/model-00002-of-00008.safetensors?download=true
wget -O model-00003-of-00008.safetensors https://hf-mirror.com/Qwen/Qwen1.5-14B-Chat/resolve/main/model-00003-of-00008.safetensors?download=true
wget -O model-00004-of-00008.safetensors https://hf-mirror.com/Qwen/Qwen1.5-14B-Chat/resolve/main/model-00004-of-00008.safetensors?download=true
wget -O model-00005-of-00008.safetensors https://hf-mirror.com/Qwen/Qwen1.5-14B-Chat/resolve/main/model-00005-of-00008.safetensors?download=true
wget -O model-00006-of-00008.safetensors https://hf-mirror.com/Qwen/Qwen1.5-14B-Chat/resolve/main/model-00006-of-00008.safetensors?download=true
wget -O model-00007-of-00008.safetensors https://hf-mirror.com/Qwen/Qwen1.5-14B-Chat/resolve/main/model-00007-of-00008.safetensors?download=true
wget -O model-00008-of-00008.safetensors https://hf-mirror.com/Qwen/Qwen1.5-14B-Chat/resolve/main/model-00008-of-00008.safetensors?download=true

# 3. 通过obs下载权重
mkdir -p /home/ma-user/work/mindformers/research/qwen1_5/14b_base
cd /home/ma-user/work/mindformers/research/qwen1_5
/home/ma-user/work/obsutil sync obs://model-data/qianwen1.5/14b/base ./14b_base

```

## 2.2 torch权重转mindspore权重

安装依赖
```bash
pip install torch transformers transformers_stream_generator einops accelerate
```

转换权重
```bash
cd /home/ma-user/work/mindformers/

python research/qwen1_5/convert_weight.py \
--torch_ckpt_dir ./research/qwen1_5/14b_base/ \
--mindspore_ckpt_path ./research/qwen1_5/14b_base/qwen15_14b_base.ckpt

```

如果报错:
ImportError: /home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch/lib/../../torch.libs/libgomp-d22c30c5.so.1.0.0: cannot allocate memory in static TLS https://gitee.com/mindspore/mindformers/issues/I9T7TQ?from=project-issueblock
执行:
```bash
# 根据错误路径做对应修改
export LD_PRELOAD='/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch.libs/libgomp-4dbbc2f2.so.1.0.0'
```

```bash
mkdir -p /home/ma-user/work/mindformers/research/qwen1_5/14b_base/rank_0/
mv /home/ma-user/work/mindformers/research/qwen1_5/14b_base/qwen15_14b_base.ckpt /home/ma-user/work/mindformers/research/qwen1_5/14b_base/rank_0/

```

# 3. 直接使用基础权重推理

```bash
export MS_GE_TRAIN=0
export MS_ENABLE_GE=1
export MS_ENABLE_REF_MODE=1

vi /home/ma-user/work/mindformers/research/qwen1_5/predict_qwen1_5_14b.yaml

```yaml
load_checkpoint: '/home/ma-user/work/mindformers/research/qwen1_5/14b_base/'
src_strategy_path_or_dir: ''
auto_trans_ckpt: True  # If true, auto transform load_checkpoint to load in distributed model
only_save_strategy: False
resume_training: False
use_parallel: True
run_mode: 'predict'

parallel_config:
  data_parallel: 1
  model_parallel: 8
  pipeline_stage: 1
  micro_batch_num: 1

model:
  model_config:
    checkpoint_name_or_path: "/home/ma-user/work/mindformers/research/qwen1_5/14b_base/rank_0/qwen15_14b_base.ckpt"

processor:
  tokenizer:
    vocab_file: "/home/ma-user/work/mindformers/research/qwen1_5/14b_base/vocab.json"
    merges_file: "/home/ma-user/work/mindformers/research/qwen1_5/14b_base/merges.txt"
```

```bash
cd /home/ma-user/work/mindformers/research
# 推理命令中参数会覆盖yaml文件中的相同参数

# 单卡推理
python qwen1_5/run_qwen1_5.py \
--config qwen1_5/predict_qwen1_5_14b.yaml \
--run_mode predict \
--use_parallel False \
--auto_trans_ckpt False \
--predict_data 帮助我制定一份去厦门的旅游攻略

# 多卡推理
bash run_singlenode.sh \
"python qwen1_5/run_qwen1_5.py \
--config qwen1_5/predict_qwen1_5_14b.yaml \
--run_mode predict \
--use_parallel True \
--load_checkpoint /home/ma-user/work/mindformers/research/qwen1_5/14b_base/ \
--auto_trans_ckpt True \
--predict_length 2048 \
--predict_data 帮助我制定一份去厦门的旅游攻略" \
/user/config/jobstart_hccl.json [0,8] 8

```

# 4. 数据准备

下载alpaca数据集

```bash
cd /home/ma-user/work/mindformers/research/qwen1_5/14b_base/

/home/ma-user/work/obsutil cp obs://model-data/qianwen/alpaca_data.json ./

```

转换数据集

```bash
cd /home/ma-user/work/mindformers/research

python qwen1_5/alpaca_converter.py \
--data_path ./qwen1_5/14b_base/alpaca_data.json \
--output_path ./qwen1_5/14b_base/alpaca-data-messages.json

```

数据预处理和Mindrecord数据生成

```bash
cd /home/ma-user/work/mindformers

python research/qwen1_5/qwen1_5_preprocess.py \
--input_glob ./research/qwen1_5/14b_base/alpaca-data-messages.json \
--vocab_file ./research/qwen1_5/14b_base/vocab.json \
--merges_file ./research/qwen1_5/14b_base/merges.txt \
--seq_length 2048 \
--output_file ./research/qwen1_5/14b_base/alpaca-messages.mindrecord

```

# 5. 模型训练

## 5.1 修改配置文件

```bash
mkdir /home/ma-user/work/mindformers/research/qwen1_5/14b_base/rank_0
mv /home/ma-user/work/mindformers/research/qwen1_5/14b_base/transform.ckpt /home/ma-user/work/mindformers/research/qwen1_5/14b_base/rank_0/
```
### 5.1.1 配置权重路径

```bash
vi /home/ma-user/work/mindformers/research/qwen1_5/finetune_qwen1_5_14b_base.yaml
```

```yaml
load_checkpoint: '/home/ma-user/work/mindformers/research/qwen1_5/14b_base/'
auto_trans_ckpt: True
train_dataset: &train_dataset
  data_loader:
    dataset_dir: "/home/ma-user/work/mindformers/research/qwen1_5/14b_base/alpaca-messages.mindrecord"
```
## 5.2 启动训练脚本

```bash
cd /home/ma-user/work/mindformers/research/qwen1_5/14b_base/

# 910B 设置如下环境变量
export MS_ASCEND_CHECK_OVERFLOW_MODE=INFNAN_MODE

bash /home/ma-user/work/mindformers/research/run_singlenode.sh \
"python /home/ma-user/work/mindformers/research/qwen1_5/run_qwen1_5.py \
--config /home/ma-user/work/mindformers/research/qwen1_5/finetune_qwen1_5_14b_base.yaml \
--load_checkpoint /home/ma-user/work/mindformers/research/qwen1_5/14b_base/ \
--use_parallel True \
--run_mode finetune \
--auto_trans_ckpt True \
--train_data /home/ma-user/work/mindformers/research/qwen1_5/14b_base/alpaca-messages.mindrecord" \
/user/config/jobstart_hccl.json [0,4] 4

```
