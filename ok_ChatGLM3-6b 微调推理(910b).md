[当前文档访问路径](https://ai-fae.readthedocs.io/zh-cn/latest/ok_InternLM-7b%20微调推理(910b).html)

- 镜像: mindspore2.2.14-cann7.0.0beta1_py_3.9-euler_2.8.3_910:v2_qwen1_5_72b
- 镜像源: swr.cn-east-292.mygaoxinai.com/huqs/mindspore2.2.14-cann7.0.0beta1_py_3.9-euler_2.8.3_910:v2_qwen1_5_72b
- 规格: Ascend: 8*ascend-d910b|CPU: 192核 1536GB

## 1.1 安装 Mindformers

```bash
git clone -b r1.0 https://gitee.com/mindspore/mindformers.git # 建议选这个版本
cd mindformers
bash build.sh

```

## 1.2 安装obsutil

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
mkdir /home/ma-user/work/mindformers/research/glm3/
cd /home/ma-user/work/mindformers/research/glm3/

# 1. 使用魔塔下载
pip install modelscope
vi down.py
#--->写入以下内容
from modelscope import snapshot_download
model_dir = snapshot_download("ZhipuAI/chatglm3-6b", revision = "v1.0.0")
#<---保存文件
python down.py

mv /home/ma-user/.cache/modelscope/hub/ZhipuAI/chatglm3-6b ./6b

# 2. 直接使用已经转换完成的预训练权重
cd /home/ma-user/work/mindformers/research/glm3/6b
obsutil cp obs://model-data/chatgml3/glm3/6b/rank_0/glm3_6b.ckpt ./
obsutil cp obs://model-data/chatgml3/glm3/6b/tokenizer.model ./

mkdir -p /home/ma-user/work/mindformers/research/glm3/6b/rank_0
mv ./glm3_6b.ckpt /home/ma-user/work/mindformers/research/glm3/6b/rank_0/

```

# 3. 数据准备

## 3.1 下载数据集

```bash
mkdir -p /home/ma-user/work/mindformers/research/glm3/6b/AdvertiseGen
cd /home/ma-user/work/mindformers/research/glm3/6b/AdvertiseGen
obsutil cp obs://model-data/chatgml3/glm3/6b/train.json ./
obsutil cp obs://model-data/chatglm32k/dev.json ./

```


# 4. 模型训练

## 4.1 RANK_TABLE_FILE

```bash
cd /home/ma-user/work/mindformers
python ./mindformers/tools/hccl_tools.py --device_num "[0,8]"
```

## 4.2 修改配置文件

```bash
cd /home/ma-user/work/mindformers/research/glm3/
cp /home/ma-user/work/mindformers/configs/glm3/run_glm3_6b_finetune_2k_910b.yaml ./
vi /home/ma-user/work/mindformers/research/glm3/run_glm3_6b_finetune_2k_910b.yaml

```

### 4.2.1 配置文件
```yaml
run_mode: 'finetune'
load_checkpoint: '/home/ma-user/work/mindformers/research/glm3/6b/'
auto_trans_ckpt: False  # If true, auto transform load_checkpoint to load in distributed model
only_save_strategy: False
resume_training: False

train_dataset: &train_dataset
  data_loader:
    dataset_dir: "/home/ma-user/work/mindformers/research/glm3/6b/AdvertiseGen/train.json"
  tokenizer:
    vocab_file: "/home/ma-user/work/mindformers/research/glm3/6b/tokenizer.model"

eval_dataset: &eval_dataset
  data_loader:
    dataset_dir: "/home/ma-user/work/mindformers/research/glm3/6b/AdvertiseGen/dev.json"
  tokenizer:
    vocab_file: "/home/ma-user/work/mindformers/research/glm3/6b/tokenizer.model"

model:
  model_config:
    seq_length: 2048 #seq_length需要等于微调数据集的max_source_length + max_target_length + 1
```

## 4.3 启动训练脚本

```bash
cd /home/ma-user/work/mindformers/scripts

bash run_distribute.sh /user/config/jobstart_hccl.json /home/ma-user/work/mindformers/research/glm3/run_glm3_6b_finetune_2k_910b.yaml '[0,8]' finetune

```

# 5. 模型推理
## 5.1 编写推理脚本
代码见: [files/generate-infer.py](https://gitee.com/lincenying/ai-fea/raw/main/files/generate-infer.py)

## 5.2 启动推理脚本
```bash
cd /home/ma-user/work/mindformers/
python research/glm3/generate-infer.py
```

# [过滤权重和权重合并](https://ai-fae.readthedocs.io/zh-cn/latest/过滤权重和权重合并.html)