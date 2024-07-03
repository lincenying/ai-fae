## 1.1 安装 Mindformers

```bash
git clone -b r1.0 https://gitee.com/mindspore/mindformers.git # 建议选这个版本
cd mindformers
bash build.sh

```

## 1.2 安装obsutil

```bash
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
cd /home/ma-user/work/mindformers/research/glm32k/

git clone https://hf-mirror.com/THUDM/chatglm3-6b-32k
rm -rf ./chatglm3-6b-32k/pytorch_model*.bin
mv chatglm3-6b-32k models

# 1. 通过huggingface权重
# 通过镜像加载huggingface权重
cd /home/ma-user/work/mindformers/research/glm32k/models
/home/ma-user/work/obsutil cp obs://model-data/chatglm3/pytorch_model-00001-of-00007.bin ./
/home/ma-user/work/obsutil cp obs://model-data/chatglm3/pytorch_model-00002-of-00007.bin ./
/home/ma-user/work/obsutil cp obs://model-data/chatglm3/pytorch_model-00003-of-00007.bin ./
/home/ma-user/work/obsutil cp obs://model-data/chatglm3/pytorch_model-00004-of-00007.bin ./
/home/ma-user/work/obsutil cp obs://model-data/chatglm3/pytorch_model-00005-of-00007.bin ./
/home/ma-user/work/obsutil cp obs://model-data/chatglm3/pytorch_model-00006-of-00007.bin ./
/home/ma-user/work/obsutil cp obs://model-data/chatglm3/pytorch_model-00007-of-00007.bin ./

# 安装可能缺失的依赖
pip install torch==1.11.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers==4.30.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tokenizers==0.15.0

cd /home/ma-user/work/mindformers

python ./research/glm32k/convert_weight.py \
--huggingface_torch_path /home/ma-user/work/mindformers/research/glm32k/models/ \
--merged_torch_path /home/ma-user/work/mindformers/research/glm32k/models/glm32k.pth \
--mindspore_path /home/ma-user/work/mindformers/research/glm32k/models/glm32k.ckpt

mkdir -p ./research/glm32k/models/rank_0
mv ./research/glm32k/models/glm32k.ckpt ./research/glm32k/models/rank_0

# 2. 直接使用已经转换完成的预训练权重
cd /home/ma-user/work/mindformers/research/glm32k/models
/home/ma-user/work/obsutil cp obs://model-data/chatglm3/glm32k.ckpt ./

mkdir -p /home/ma-user/work/mindformers/research/glm32k/models/rank_0
mv ./glm32k.ckpt ./research/glm32k/models/rank_0

```

# 3. 数据准备

## 3.1 创建数据存放目录
 
 ```bash
cd /home/ma-user/work/mindformers/research/glm32k/models
mkdir AdvertiseGen
cd AdvertiseGen
/home/ma-user/work/obsutil cp obs://model-data/chatglm3/train.json ./
/home/ma-user/work/obsutil cp obs://model-data/chatglm3/dev.json ./
cd ..

```

## 3.2 下载原始数据

```bash
cd /home/ma-user/work/mindformers/research/glm32k/models
# LongBench 数据集
/home/ma-user/work/obsutil cp obs://model-data/chatglm3/data.zip ./
unzip data.zip
wget https://raw.githubusercontent.com/THUDM/LongBench/main/config/dataset2prompt.json

```

## 3.3 数据格式转换

```bash
cd /home/ma-user/work/mindformers/research/glm32k

python glm32k_preprocess.py \
--data_path ./models/data \
--output_path ./models \
--prompt_config_file ./models/dataset2prompt.json

```

# 4. 模型训练

## 4.1 RANK_TABLE_FILE

```bash
cd /home/ma-user/work/mindformers
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)"
```

## 4.2 修改配置文件

```bash
cd /home/ma-user/workmindformers/research/glm32k/
vi /home/ma-user/work/mindformers/research/glm32k/run_glm32k.yaml

```

### 4.2.1 配置文件
```yaml
run_mode: 'finetune'
output_dir: './output' # path to save checkpoint/strategy
load_checkpoint: '/home/ma-user/work/mindformers/research/glm32k/models/'
auto_trans_ckpt: False  # If true, auto transform load_checkpoint to load in distributed model
only_save_strategy: False
resume_training: False

train_dataset: &train_dataset
  data_loader:
    type: ADGenDataLoader
    dataset_dir: "/home/ma-user/work/mindformers/research/glm32k/models/AdvertiseGen/train.json"
    shuffle: True
    phase: "train"
    version: 3
    origin_columns: ["content", "summary"]
  tokenizer:
    type: ChatGLM32kTokenizer
    vocab_file: "/home/ma-user/work/mindformers/research/glm32k/models/tokenizer.model"
  input_columns: ["input_ids", "labels"]
  max_source_length: 30720
  max_target_length: 2047

eval_dataset: &eval_dataset
  data_loader:
    type: ADGenDataLoader
    dataset_dir: "/home/ma-user/work/mindformers/research/glm32k/models/AdvertiseGen/dev.json"
    shuffle: False
    phase: "eval"
    version: 2
    origin_columns: ["content", "summary"]
  tokenizer:
    type: ChatGLM32kTokenizer
    vocab_file: "/home/ma-user/work/mindformers/research/glm32k/models/tokenizer.model"

model:
  model_config:
    seq_length: 32768
    pre_seq_len: 0
```

注意: pre_seq_len 必须设置为0, 不然会报错

## 4.3 启动训练脚本

```bash
cd /home/ma-user/work/mindformers/research

bash run_singlenode.sh \
"python glm32k/run_glm32k.py \
--config glm32k/run_glm32k.yaml \
--load_checkpoint /home/ma-user/work/mindformers/research/glm32k/models/ \
--use_parallel True \
--run_mode finetune \
--train_data /home/ma-user/work/mindformers/research/glm32k/models/longbench.jsonl" \
/user/config/jobstart_hccl.json [0,8] 8

```

报错: https://gitee.com/mindspore/mindformers/issues/I9G1GJ?from=project-issue

python infer_generate.py --checkpoint_path /home/ma-user/work/mindformers/research/glm32k/models/rank_0/glm32k.ckpt --device_id 0 --user_query "晚上睡不着应该怎么办"