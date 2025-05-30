[当前文档访问路径](https://ai-fae.readthedocs.io/zh-cn/latest/ok_QWen1.5-72b%20推理(910b).html)

- 镜像: mindspore2.2.14-cann7.0.0beta1_py_3.9-euler_2.8.3_910:v2_qwen1_5_72b
- 镜像源: swr.cn-east-292.mygaoxinai.com/huqs/mindspore2.2.14-cann7.0.0beta1_py_3.9-euler_2.8.3_910:v2_qwen1_5_72b
- 规格: Ascend: 8*ascend-d910b|CPU: 192核 1536GB


## 1.1 安装 Mindformers

```bash
git clone -b r1.0 https://gitee.com/mindspore/mindformers.git # 建议选这个版本
cd mindformers
bash build.sh

```

# 2. 权重准备

## 2.1 创建权重存放目录及下载
```bash
cd /home/ma-user/work/mindformers/research/qwen1_5


# 通过obs下载权重
cd /home/ma-user/work/
# 下载obsutil
wget https://obs-community.obs.cn-north-1.myhuaweicloud.com/obsutil/current/obsutil_linux_arm64.tar.gz
# 解压缩obsutil
tar -zxvf obsutil_linux_arm64.tar.gz
# 修改可执行文件
chmod +x ./obsutil_linux_arm64_5.7.3/obsutil
# 移动obsutil
mv ./obsutil_linux_arm64_5.7.3 ./obs_bin
# 添加环境变量
export OBSAK="这里改成AK"
export OBSSK="这里改成SK"
# notebook停止后也需要重新执行下面两条命令
export PATH=$PATH:/home/ma-user/work/obs_bin
obsutil config -i=${OBSAK} -k=${OBSSK} -e=obs.cn-east-292.mygaoxinai.com

```
### 2.1.1 直接使用转换完成的权重

```bash
mkdir -p /home/ma-user/work/mindformers/research/qwen1_5/72b/rank_0/
cd /home/ma-user/work/mindformers/research/qwen1_5/72b/rank_0/

obsutil cp obs://wio/qw1.5-72b-chat.ckpt ./qwen1_5-72b.ckpt
```

### 2.1.2 使用huggingface权重自行转换

```bash
cd /home/ma-user/work/mindformers/research/qwen1_5/
obsutil sync obs://bigmodel/qwen1.5-72b-chat/ ./qwen1.5-72b-chat
mv qwen1.5-72b-chat/ 72b/

```

#### 2.1.2.1 torch权重转mindspore权重

安装依赖
```bash
pip install torch transformers transformers_stream_generator einops accelerate

```

转换权重
```bash
cd /home/ma-user/work/mindformers/

python research/qwen1_5/convert_weight.py \
--torch_ckpt_dir ./research/qwen1_5/72b/ \
--mindspore_ckpt_path ./research/qwen1_5/72b/qwen1_5-72b.ckpt

```

如果报错:
ImportError: /home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch/lib/../../torch.libs/libgomp-d22c30c5.so.1.0.0: cannot allocate memory in static TLS block
执行:
```bash
# libgomp-d22c30c5.so.1.0.0 文件名可能会不同, 把下面路径替换成实际路径即可
export LD_PRELOAD='/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch.libs/libgomp-d22c30c5.so.1.0.0'
```
命令后, 重新执行转换权重命令

权重转换成功后, 继续执行下面的命令

```bash
mkdir -p /home/ma-user/work/mindformers/research/qwen1_5/72b/rank_0/
mv /home/ma-user/work/mindformers/research/qwen1_5/72b/qwen1_5-72b.ckpt /home/ma-user/work/mindformers/research/qwen1_5/72b/rank_0/qwen1_5-72b.ckpt
```

# 3. 直接使用基础权重推理

```bash
# Atlas 800T A2上运行时需要设置如下环境变量，否则推理结果会出现精度问题
export MS_GE_TRAIN=0
export MS_ENABLE_GE=1
export MS_ENABLE_REF_MODE=1

vi /home/ma-user/work/mindformers/research/qwen1_5/run_qwen1_5_72b_infer.yaml
```

```yaml
load_checkpoint: '/home/ma-user/work/mindformers/research/qwen1_5/72b/'
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
    vocab_file: "/home/ma-user/work/mindformers/research/qwen1_5/72b/vocab.json"
    merges_file: "/home/ma-user/work/mindformers/research/qwen1_5/72b/merges.txt"
```

```bash
cd /home/ma-user/work/mindformers/research
# 推理命令中参数会覆盖yaml文件中的相同参数

# 多卡推理
bash run_singlenode.sh \
"python qwen1_5/run_qwen1_5.py \
--config qwen1_5/run_qwen1_5_72b_infer.yaml \
--run_mode predict \
--use_parallel True \
--load_checkpoint /home/ma-user/work/mindformers/research/qwen1_5/72b/ \
--auto_trans_ckpt True \
--predict_data 帮助我制定一份去杭州的旅游攻略" \
/user/config/jobstart_hccl.json [0,4] 4

```

