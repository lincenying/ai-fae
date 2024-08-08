[当前文档访问路径](https://ai-fae.readthedocs.io/zh-cn/latest/ok_Baichuan2-7b%20%E5%BE%AE%E8%B0%83%E6%8E%A8%E7%90%86(910a).html)

910, 镜像 ms2_2_0_cann_7_0_py39:v4

# 1. 环境准备

```bash
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Ascend%20HDK/Ascend%20HDK%2023.0.3/Ascend-hdk-910b-npu-driver_23.0.3_linux-aarch64.run
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Ascend%20HDK/Ascend%20HDK%2023.0.3/Ascend-hdk-910b-npu-firmware_7.1.0.5.220.run

wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.0/Ascend-cann-toolkit_7.0.0_linux-aarch64.run
chmod +x Ascend-cann-toolkit_7.0.0_linux-aarch64.run
./Ascend-cann-toolkit_7.0.0_linux-aarch64.run --install --install-for-all

wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.0/Ascend-cann-kernels-910b_7.0.0_linux.run
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.0/Ascend-cann-kernels-910b_7.0.0_linux.run.p7s

```

## 1.0 安装 MindSpore 2.2.14

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.14/MindSpore/unified/aarch64/mindspore-2.2.14-cp39-cp39-linux_aarch64.whl \
--trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com \
-i https://pypi.tuna.tsinghua.edu.cn/simple
```

↑↑↑↑↑↑↑↑ModelArt省略上面步骤↑↑↑↑↑↑↑↑↑↑

## 1.1 安装 Mindformers

```bash
git clone -b r1.1.rc1 https://gitee.com/mindspore/mindformers.git # 建议选这个版本
git clone -b dev https://gitee.com/mindspore/mindformers.git # 开发版容易出问题
cd mindformers
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
cd /home/ma-user/work/mindformers/research/baichuan2/

git clone https://hf-mirror.com/baichuan-inc/Baichuan2-7B-Chat
rm -rf ./Baichuan2-7B-Chat/pytorch_model.bin
rm -rf ./Baichuan2-7B-Chat/tokenizer.model
mv Baichuan2-7B-Chat models

# 1. 通过huggingface权重
# 通过镜像加载huggingface权重
# wget -O pytorch_model.bin https://hf-mirror.com/baichuan-inc/Baichuan2-7B-Chat/resolve/main/pytorch_model.bin?download=true
cd /home/ma-user/work/mindformers/research/baichuan2/models
obsutil cp obs://model-data/baichuan2/pytorch_model.bin ./



cd /home/ma-user/work/mindformers

# 安装可能缺失的依赖
pip install torch==2.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers==4.35.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tokenizers==0.15.0
pip install xformers

python ./research/baichuan/convert_weight.py \
--torch_ckpt_path /home/ma-user/work/mindformers/research/baichuan2/models/pytorch_model.bin \
--mindspore_ckpt_path /home/ma-user/work/mindformers/research/baichuan2/models/pytorch_model.ckpt

mkdir -p /home/ma-user/work/mindformers/research/baichuan2/models/rank_0
mv /home/ma-user/work/mindformers/research/baichuan2/models/pytorch_model.ckpt /home/ma-user/work/mindformers/research/baichuan2/models/rank_0

# ImportError: /home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch/lib/../../torch.libs/libgomp-d22c30c5.so.1.0.0: cannot allocate memory in static TLS block
# 如果报上面错误, 执行下面命令
export LD_PRELOAD=$LD_PRELOAD:/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch.libs/libgomp-d22c30c5.so.1.0.0

# 2. 直接使用已经转换完成的预训练权重
cd /home/ma-user/work/mindformers/research/baichuan2/models
# wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/Baichuan2_7B_Chat.ckpt
obsutil cp obs://model-data/baichuan2/Baichuan2_7B_Chat.ckpt ./

mkdir -p /home/ma-user/work/mindformers/research/baichuan2/models/rank_0
mv /home/ma-user/work/mindformers/research/baichuan2/models/Baichuan2_7B_Chat.ckpt /home/ma-user/work/mindformers/research/baichuan2/models/rank_0

```
 
## 2.2 分词器文件下载
```bash
cd /home/ma-user/work/mindformers/research/baichuan2/models
# wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/tokenizer.model
obsutil cp obs://model-data/baichuan2/tokenizer.model ./

```

# 3. 数据准备

## 3.1 创建数据存放目录
 
 ```bash
cd /home/ma-user/work/mindformers/research/baichuan2/models
```

## 3.2 下载原始数据

```bash
# wget https://github.com/baichuan-inc/Baichuan2/raw/main/fine-tune/data/belle_chat_ramdon_10k.json
obsutil cp obs://model-data/baichuan2/belle_chat_ramdon_10k.json ./
```

## 3.3 数据格式转换

```bash
cd /home/ma-user/work/mindformers/

python research/baichuan2/belle_preprocess.py \
--input_glob /home/ma-user/work/mindformers/research/baichuan2/models/belle_chat_ramdon_10k.json \
--model_file /home/ma-user/work/mindformers/research/baichuan2/models/tokenizer.model \
--output_file /home/ma-user/work/mindformers/research/baichuan2/models/belle_512.mindrecord \
--seq_length 512

```

# 4. 模型训练

## 4.1 RANK_TABLE_FILE

```bash
cd /home/ma-user/work/mindformers
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)"
```

## 4.1 修改配置文件

```bash
vi /home/ma-user/work/mindformers/research/baichuan2/run_baichuan2_7b_910b.yaml
```

### 4.1 配置文件
```yaml
seed: 0
output_dir: './output' # path to save checkpoint/strategy
load_checkpoint: '/home/ma-user/work/mindformers/research/baichuan2/models/rank_0/Baichuan2_7B_Chat.ckpt'
src_strategy_path_or_dir: ''
auto_trans_ckpt: True
only_save_strategy: False
resume_training: False
use_parallel: True
run_mode: 'finetune'

train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "/home/ma-user/work/mindformers/research/baichuan2/models/belle_512.mindrecord"
    shuffle: True
  input_columns: ["input_ids", "labels"]  # "input_ids", "labels" , labels are used in instruction finetune.
  num_parallel_workers: 8
  python_multiprocessing: False
  drop_remainder: True
  repeat: 1
  numa_enable: False
  prefetch_size: 1

model:
  model_config:
    checkpoint_name_or_path: "/home/ma-user/work/mindformers/research/baichuan2/models/rank_0/Baichuan2_7B_Chat.ckpt"
```

## 4.2 启动训练脚本

```bash
cd /home/ma-user/work/mindformers/research

bash run_singlenode.sh \
"python baichuan2/run_baichuan2.py \
--config baichuan2/run_baichuan2_7b_910b.yaml \
--load_checkpoint /home/ma-user/work/mindformers/research/baichuan2/models/rank_0/Baichuan2_7B_Chat.ckpt \
--auto_trans_ckpt True \
--use_parallel True \
--run_mode finetune \
--train_dataset /home/ma-user/work/mindformers/research/baichuan2/models/belle_512.mindrecord" \
/user/config/jobstart_hccl.json [0,8] 8

```

训练进程后台运行，可在`/home/ma-user/work/mindformers/research/output/log/rank_0` 目录下查看实时更新的训练日志

# 5. 模型推理

## 5.1 过滤权重参数

```bash
cd /home/ma-user/work/mindformers/research/output
vi filter_ckpt_param.py
# 内容见files文件夹下
```

```bash
python ./filter_ckpt_param.py
```

## 5.2 合并权重

```bash
vi transform_ckpt.py
# 内容见files文件夹下
```

## 5.3 修改配置文件

```bash
vi /home/ma-user/work/mindformers/research/baichuan2/run_baichuan2_7b.yaml
```

修改权重初始化类型，将默认的 ”float16” 改为 ”float32”，保障推理精度(非必须，精度要求高场 景下可开启)

```yaml
seed: 0
output_dir: './output' # path to save checkpoint/strategy
load_checkpoint: 'baichuan2/models/pytorch_model.ckpt'
src_strategy_path_or_dir: ''
auto_trans_ckpt: False  # If true, auto transform load_checkpoint to load in distributed model
only_save_strategy: False
resume_training: False
use_parallel: False
run_mode: 'train'

model:
  model_config:
    param_init_type: "float32"                          # 可选
    use_past: True                                      # 使用增量推理
processor:
  return_tensors: ms
  tokenizer:
    unk_token: '<unk>'
    bos_token: '<s>'
    eos_token: '</s>'
    pad_token: '<unk>'
    type: Baichuan2Tokenizer
    vocab_file: 'baichuan2/models/tokenizer.model'
  type: LlamaProcessor
```

## 5.4 运行推理脚本

```bash
# 转换后的初始权重
cd /home/ma-user/work/mindformers/research/

python baichuan2/run_baichuan2.py \
--config baichuan2/run_baichuan2_7b.yaml \
--run_mode predict \
--use_parallel False \
--load_checkpoint baichuan2/models/pytorch_model.ckpt \
--auto_trans_ckpt False \
--predict_data "<reserved_106>你是谁？<reserved_107>"

# 预训练权重
cd /home/ma-user/work/mindformers/research/

python baichuan2/run_baichuan2.py \
--config baichuan2/run_baichuan2_7b.yaml \
--run_mode predict \
--use_parallel False \
--load_checkpoint baichuan2/models/Baichuan2_7B_Chat.ckpt \
--auto_trans_ckpt False \
--predict_data "<reserved_106>你是谁？<reserved_107>"

# 微调过后的权重
cd /home/ma-user/work/mindformers/research/

python baichuan2/run_baichuan2.py \
--config baichuan2/run_baichuan2_7b.yaml \
--run_mode predict \
--use_parallel False \
--load_checkpoint ./output/ckpt/rank_0/checkpoint_0.ckpt \
--auto_trans_ckpt False \
--predict_data "<reserved_106>你是谁？<reserved_107>"

```

如果报错: AttributeError: Baichuan2Tokenizer: 'tokenizers.AddedToken' object has no attribute 'special'

```bash
pip install tokenizers==0.15.0

```