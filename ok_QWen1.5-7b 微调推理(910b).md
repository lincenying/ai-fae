[当前文档访问路径](https://ai-fae.readthedocs.io/zh-cn/latest/ok_QWen1.5-7b%20微调推理(910b).html)

- 镜像: mindspore2.2.14-cann7.0.0beta1_py_3.9-euler_2.8.3_910:v2_qwen1_5_72b
- 镜像源: swr.cn-east-292.mygaoxinai.com/huqs/mindspore2.2.14-cann7.0.0beta1_py_3.9-euler_2.8.3_910:v2_qwen1_5_72b
- 规格: Ascend: 8*ascend-d910b|CPU: 192核 1536GB


## 1.1 安装 Mindformers

```bash
git clone -b r1.0 https://gitee.com/mindspore/mindformers.git
cd mindformers
git branch # 确认分支是r1.0  如果不是, 执行 git checkout r1.0
bash build.sh


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
cd /home/ma-user/work/mindformers/research/qwen1_5
```
### 2.1.1 直接使用转换完成的权重

```bash
git clone https://hf-mirror.com/Qwen/Qwen1.5-7B-Chat 7b_chat
# 删除无效的权重
rm -rf ./7b_chat/*.safetensors

mkdir -p /home/ma-user/work/mindformers/research/qwen1_5/7b_chat/rank_0/
cd /home/ma-user/work/mindformers/research/qwen1_5/7b_chat/rank_0/

obsutil cp obs://model-data/qianwen1.5/7b/chat/qwen15_7b_chat.ckpt ./qwen15_7b_chat.ckpt

```

### 2.1.2 使用huggingface权重自行转换

#### 2.1.2.1 下载huggingface权重

```bash
# 1. 使用魔搭下载
pip install modelscope
cd /home/ma-user/work/mindformers/research/qwen1_5

modelscope download --model 'qwen/Qwen1.5-7B-Chat' --local_dir './7b_chat'

# 2. 通过hf-mirror克隆文件
git clone https://hf-mirror.com/Qwen/Qwen1.5-7B-Chat 7b_chat
# 删除无效的权重
rm -rf ./7b_chat/*.safetensors
cd 7b_chat

# 2.1 使用hf-mirror下载权重
wget -O model-00001-of-00008.safetensors https://hf-mirror.com/Qwen/Qwen1.5-7B-Chat/resolve/main/model-00001-of-00008.safetensors?download=true
wget -O model-00002-of-00008.safetensors https://hf-mirror.com/Qwen/Qwen1.5-7B-Chat/resolve/main/model-00002-of-00008.safetensors?download=true
wget -O model-00003-of-00008.safetensors https://hf-mirror.com/Qwen/Qwen1.5-7B-Chat/resolve/main/model-00003-of-00008.safetensors?download=true
wget -O model-00004-of-00008.safetensors https://hf-mirror.com/Qwen/Qwen1.5-7B-Chat/resolve/main/model-00004-of-00008.safetensors?download=true

# 2.2 通过obs下载权重

obsutil cp obs://model-data/qianwen1.5/7b/chat/model-00001-of-00004.safetensors ./
obsutil cp obs://model-data/qianwen1.5/7b/chat/model-00002-of-00004.safetensors ./
obsutil cp obs://model-data/qianwen1.5/7b/chat/model-00003-of-00004.safetensors ./
obsutil cp obs://model-data/qianwen1.5/7b/chat/model-00004-of-00004.safetensors ./

```

#### 2.1.2.2 torch权重转mindspore权重

安装依赖
```bash
pip install torch transformers transformers_stream_generator einops accelerate
```

转换权重
```bash
cd /home/ma-user/work/mindformers/

python research/qwen1_5/convert_weight.py \
--torch_ckpt_dir ./research/qwen1_5/7b_chat/ \
--mindspore_ckpt_path ./research/qwen1_5/7b_chat/qwen15_7b_chat.ckpt

```

如果报错:
ImportError: /home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch/lib/../../torch.libs/libgomp-d22c30c5.so.1.0.0: cannot allocate memory in static TLS block
执行:
```bash
# libgomp-d22c30c5.so.1.0.0  文件名可能不一样, 按照报错提示, 修改下面的路径
export LD_PRELOAD='/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch.libs/libgomp-d22c30c5.so.1.0.0'
```
命令后, 重新执行转换权重命令

权重转换成功后, 继续执行下面的命令

```bash
mkdir -p /home/ma-user/work/mindformers/research/qwen1_5/7b_chat/rank_0/
mv ./research/qwen1_5/7b_chat/qwen15_7b_chat.ckpt /home/ma-user/work/mindformers/research/qwen1_5/7b_chat/rank_0/

```

# 3. 直接使用基础权重推理

```bash
export MS_GE_TRAIN=0
export MS_ENABLE_GE=1
export MS_ENABLE_REF_MODE=1

vi /home/ma-user/work/mindformers/research/qwen1_5/run_qwen1_5_7b_infer.yaml
```

将 https://gitee.com/lincenying/ai-fea/raw/main/config/run_qwen1_5_7b_infer.yaml 里的内容复制进去后, 按下面的内容替换对应字段

```yaml
load_checkpoint: '/home/ma-user/work/mindformers/research/qwen1_5/7b_chat/'
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

processor:
  tokenizer:
    vocab_file: "/home/ma-user/work/mindformers/research/qwen1_5/7b_chat/vocab.json"
    merges_file: "/home/ma-user/work/mindformers/research/qwen1_5/7b_chat/merges.txt"
```

```bash
cd /home/ma-user/work/mindformers/research
# 推理命令中参数会覆盖yaml文件中的相同参数

# 单卡推理
python qwen1_5/run_qwen1_5.py \
--config qwen1_5/run_qwen1_5_7b_infer.yaml \
--run_mode predict \
--use_parallel False \
--load_checkpoint /home/ma-user/work/mindformers/research/qwen1_5/7b_chat/rank_0/qwen15_7b_chat.ckpt \
--auto_trans_ckpt False \
--predict_length 2048 \
--predict_data 帮助我制定一份去杭州的旅游攻略

# 多卡推理
bash run_singlenode.sh \
"python qwen1_5/run_qwen1_5.py \
--config qwen1_5/run_qwen1_5_7b_infer.yaml \
--run_mode predict \
--use_parallel True \
--load_checkpoint /home/ma-user/work/mindformers/research/qwen1_5/7b_chat/ \
--auto_trans_ckpt True \
--predict_length 2048 \
--predict_data 帮助我制定一份去杭州的旅游攻略" \
/user/config/jobstart_hccl.json [0,4] 4

```

# 4. 数据准备

下载alpaca数据集

```bash
cd /home/ma-user/work/mindformers/research/qwen1_5/7b_chat/

obsutil cp obs://model-data/qianwen/alpaca_data.json ./

wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/qwen/qwen.tiktoken

```

转换数据集

```bash
cd /home/ma-user/work/mindformers/research

python qwen1_5/alpaca_converter.py \
--data_path ./qwen1_5/7b_chat/alpaca_data.json \
--output_path ./qwen1_5/7b_chat/alpaca-data-messages.json

```

数据预处理和Mindrecord数据生成

```bash
cd /home/ma-user/work/mindformers

python research/qwen1_5/qwen1_5_preprocess.py \
--input_glob ./research/qwen1_5/7b_chat/alpaca-data-messages.json \
--vocab_file ./research/qwen1_5/7b_chat/vocab.json \
--merges_file ./research/qwen1_5/7b_chat/merges.txt \
--seq_length 2048 \
--output_file ./research/qwen1_5/7b_chat/alpaca-messages.mindrecord

```

# 5. 模型微调训练

## 5.1 修改配置文件

### 5.1.1 配置权重路径

```bash
vi /home/ma-user/work/mindformers/research/qwen1_5/run_qwen1_5_7b.yaml
```

将 https://gitee.com/lincenying/ai-fea/raw/main/config/run_qwen1_5_7b.yaml 里的内容复制进去后, 按下面的内容替换对应字段

```yaml
load_checkpoint: '/home/ma-user/work/mindformers/research/qwen1_5/7b_chat/'
auto_trans_ckpt: True
train_dataset: &train_dataset
  data_loader:
    dataset_dir: "/home/ma-user/work/mindformers/research/qwen1_5/7b_chat/alpaca-messages.mindrecord"

runner_config:
  epochs: 5 # 训练轮数
  batch_size: 1
  sink_mode: True
  sink_size: 1

parallel_config:
  data_parallel: 4
  model_parallel: 1
  pipeline_stage: 1
  micro_batch_num: 1
  gradient_aggregation_group: 4
micro_batch_interleave_num: 1
```
data_parallel * model_parallel * pipeline_stage = 卡数

## 5.2 启动微调训练脚本

```bash
export MS_ASCEND_CHECK_OVERFLOW_MODE=INFNAN_MODE
# 如出现OOM需要配置:
 # 打开内存复用
export ENABLE_CELL_RESUSE=1
# 打开内存优化     
export MS_GE_ATOMIC_CLEAN_POLICY=1

cd /home/ma-user/work/mindformers/research/qwen1_5/7b_chat/

bash /home/ma-user/work/mindformers/research/run_singlenode.sh \
"python /home/ma-user/work/mindformers/research/qwen1_5/run_qwen1_5.py \
--config /home/ma-user/work/mindformers/research/qwen1_5/run_qwen1_5_7b.yaml \
--load_checkpoint /home/ma-user/work/mindformers/research/qwen1_5/7b_chat/rank_0/qwen15_7b_chat.ckpt \
--use_parallel True \
--run_mode finetune \
--auto_trans_ckpt True \
--predict_length 2048 \
--train_data /home/ma-user/work/mindformers/research/qwen1_5/7b_chat/alpaca-messages.mindrecord" \
/user/config/jobstart_hccl.json [0,4] 4

```

# [过滤权重和权重合并](https://ai-fae.readthedocs.io/zh-cn/latest/过滤权重和权重合并.html)