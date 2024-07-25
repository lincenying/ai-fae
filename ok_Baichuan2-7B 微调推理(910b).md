- 镜像: mindspore2.2.10-cann7.0.0beta1_py_3.9-euler_2.8.3_910b:v2
- 镜像源: swr.cn-east-292.mygaoxinai.com/huqs/mindspore2.2.10-cann7.0.0beta1_py_3.9-euler_2.8.3_910b:v2
- Ascend: 8*ascend-d910b|CPU: 192核 1536GB

↓↓↓↓↓↓↓↓↓ModelArt省略下面步骤↓↓↓↓↓↓↓↓↓↓

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

## 1.0 安装 MindSpore 2.2.10

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.10/MindSpore/unified/aarch64/mindspore-2.2.10-cp39-cp39-linux_aarch64.whl \
--trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com \
-i https://pypi.tuna.tsinghua.edu.cn/simple
```

↑↑↑↑↑↑↑↑ModelArt省略上面步骤↑↑↑↑↑↑↑↑↑↑

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
cd /home/ma-user/work/mindformers/research/baichuan2/

git clone https://hf-mirror.com/baichuan-inc/Baichuan2-7B-Chat
rm -rf ./Baichuan2-7B-Chat/pytorch_model.bin
rm -rf ./Baichuan2-7B-Chat/tokenizer.model
mv Baichuan2-7B-Chat 7b

# 1. 通过huggingface权重
# 通过镜像加载huggingface权重
# wget -O pytorch_model.bin https://hf-mirror.com/baichuan-inc/Baichuan2-7B-Chat/resolve/main/pytorch_model.bin?download=true
# wget -O pytorch_model.bin https://hf-mirror.com/baichuan-inc/Baichuan2-13B-Chat/resolve/main/pytorch_model.bin?download=true
cd /home/ma-user/work/mindformers/research/baichuan2/7b
/home/ma-user/work/obsutil cp obs://model-data/baichuan2/pytorch_model.bin ./


cd /home/ma-user/work/mindformers

# 安装可能缺失的依赖
pip install torch==2.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers==4.35.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tokenizers==0.15.0
pip install xformers

python ./research/baichuan/convert_weight.py \
--torch_ckpt_path /home/ma-user/work/mindformers/research/baichuan2/7b/pytorch_model.bin \
--mindspore_ckpt_path /home/ma-user/work/mindformers/research/baichuan2/7b/pytorch_model.ckpt

mkdir -p ./research/baichuan2/7b/rank_0
mv ./research/baichuan2/7b/pytorch_model.ckpt ./research/baichuan2/7b/rank_0

# ImportError: /home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch/lib/../../torch.libs/libgomp-d22c30c5.so.1.0.0: cannot allocate memory in static TLS block
# 如果报上面错误, 执行下面命令
export LD_PRELOAD=$LD_PRELOAD:/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch.libs/libgomp-d22c30c5.so.1.0.0

# 2. 直接使用已经转换完成的预训练权重
cd /home/ma-user/work/mindformers/research/baichuan2/7b

# wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/Baichuan2_7B_Chat.ckpt
/home/ma-user/work/obsutil cp obs://model-data/baichuan2/Baichuan2_7B_Chat.ckpt ./

mkdir -p /home/ma-user/work/mindformers/research/baichuan2/7b/rank_0
mv /home/ma-user/work/mindformers/research/baichuan2/7b/Baichuan2_7B_Chat.ckpt /home/ma-user/work/mindformers/research/baichuan2/7b/rank_0
```
 
## 2.2 分词器文件下载
```bash
cd /home/ma-user/work/mindformers/research/baichuan2/7b
# wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/baichuan2/tokenizer.model
/home/ma-user/work/obsutil cp obs://model-data/baichuan2/tokenizer.model ./

```

# 3. 数据准备

## 3.1 创建数据存放目录
 
 ```bash
cd /home/ma-user/work/mindformers/research/baichuan2/7b
```

## 3.2 下载原始数据

```bash
# wget https://github.com/baichuan-inc/Baichuan2/raw/main/fine-tune/data/belle_chat_ramdon_10k.json
/home/ma-user/work/obsutil cp obs://model-data/baichuan2/belle_chat_ramdon_10k.json ./
```

## 3.3 数据格式转换

```bash
cd /home/ma-user/work/mindformers/

python research/baichuan2/belle_preprocess.py \
--input_glob /home/ma-user/work/mindformers/research/baichuan2/7b/belle_chat_ramdon_10k.json \
--model_file /home/ma-user/work/mindformers/research/baichuan2/7b/tokenizer.model \
--output_file /home/ma-user/work/mindformers/research/baichuan2/7b/belle_512.mindrecord \
--seq_length 512

```

# 4. 模型微调训练

## 4.1 RANK_TABLE_FILE

```bash
cd /home/ma-user/work/mindformers
python ./mindformers/tools/hccl_tools.py --device_num "[0,8)"
```

## 4.1 修改配置文件

```bash
vi /home/ma-user/work/mindformers/research/baichuan2/run_baichuan2_7b_lora_910b.yaml
```

### 4.1 配置文件
```yaml
seed: 0
output_dir: './output' # path to save checkpoint/strategy
load_checkpoint: '/home/ma-user/work/mindformers/research/baichuan2/7b/'
src_strategy_path_or_dir: ''
auto_trans_ckpt: True
only_save_strategy: False
resume_training: False
use_parallel: True
run_mode: 'finetune'

train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: "/home/ma-user/work/mindformers/research/baichuan2/7b/belle_512.mindrecord"
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
    ### 7b
    checkpoint_name_or_path: "/home/ma-user/work/mindformers/research/baichuan2/7b/rank_0/Baichuan2_7B_Chat.ckpt"
    ### 13b
    checkpoint_name_or_path: "/home/ma-user/work/mindformers/research/baichuan2/7b/rank_0/Baichuan2_13B_Chat.ckpt"
```

## 4.2 启动微调训练脚本

```bash
cd /home/ma-user/work/mindformers/research/baichuan2/7b/

export MS_ASCEND_CHECK_OVERFLOW_MODE=INFNAN_MODE

bash /home/ma-user/work/mindformers/research/run_singlenode.sh \
"python /home/ma-user/work/mindformers/research/baichuan2/run_baichuan2.py \
--config /home/ma-user/work/mindformers/research/baichuan2/run_baichuan2_7b_lora_910b.yaml \
--load_checkpoint /home/ma-user/work/mindformers/research/baichuan2/7b/ \
--auto_trans_ckpt True \
--use_parallel True \
--run_mode finetune \
--train_dataset /home/ma-user/work/mindformers/research/baichuan2/7b/belle_512.mindrecord" \
/user/config/jobstart_hccl.json [0,4] 4

```

训练进程后台运行，可在`/home/ma-user/work/mindformers/research/output/log/rank_0` 目录下查看实时更新的训练日志

# 5. 模型推理

## 5.1 过滤权重参数

```bash
cd /home/ma-user/work/mindformers/research/output
vi filter_ckpt_param.py
```

```py
import os
from glob import glob
import mindspore as ms

ignore_keys = ['accu_grads',
               'scale_sense',
               'global_step',
               'adam',
               'current_iterator_step',
               'last_overflow_iterator_step',
               'epoch_num',
               'step_num',
               'loss_scale']

def only_save_model_param(ckpt_path, save_path):
    checkpoint = ms.load_checkpoint(ckpt_path)
    new_param_list = []
    for name, param in checkpoint.items():
        ignore = False
        for key in ignore_keys:
            if key in name:
                ignore = True
                break
        if not ignore:
            new_param_list.append({"name": name, "data": param})
    ms.save_checkpoint(new_param_list, save_path)
    print(f"process {ckpt_path} finished!")
    
if __name__ == '__main__':
    
    ckpt_path_or_dir = '/home/ma-user/work/mindformers/research/output/checkpoint_network'
    assert os.path.exists(ckpt_path_or_dir), f'{ckpt_path_or_dir} not exists!' 
    if os.path.isfile(ckpt_path_or_dir):
        ckpt_paths = [ckpt_path_or_dir]
    elif os.path.isdir(ckpt_path_or_dir):
        ckpt_paths = glob(os.path.join(ckpt_path_or_dir, 'rank*/*.ckpt'))
    
    save_root = "filter_out"
    for ckpt_path in ckpt_paths:
        replace_part = ckpt_path.split('/rank')[0]
        save_path = ckpt_path.replace(replace_part, save_root)
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        only_save_model_param(ckpt_path, save_path)
```

```bash
python ./filter_ckpt_param.py

vi transform_ckpt.py
```

## 5.2 合并权重

```py
import os
import argparse
import mindspore as ms

def get_strategy(startegy_path, rank_id=None):
    """Merge strategy if strategy path is dir

    Args:
        startegy_path (str): The path of stategy.
        rank_id (int): The rank id of device.

    Returns:
        None or strategy path
    """
    if not startegy_path:
        return None

    assert os.path.exists(startegy_path), f'{startegy_path} not found!'

    if os.path.isfile(startegy_path):
        return startegy_path

    if os.path.isdir(startegy_path):
        if rank_id:
            merge_path = os.path.join(startegy_path, f'merged_ckpt_strategy_{rank_id}.ckpt')
        else:
            merge_path = os.path.join(startegy_path, f'merged_ckpt_strategy.ckpt')

        if os.path.exists(merge_path):
            os.remove(merge_path)

        ms.merge_pipeline_strategys(startegy_path, merge_path)
        return merge_path

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_ckpt_strategy',
                        default="/home/ma-user/work/mindformers/research/output/strategy",
                        help='path of src ckpt strategy')
    parser.add_argument('--dst_ckpt_strategy',
                        default="",
                        help='path of dst ckpt strategy')
    parser.add_argument('--src_ckpt_dir',
                        default="/home/ma-user/work/mindformers/research/output/transformed_checkpoint/Baichuan2_7B_Chat",
                        type=str,
                        help='path of src ckpt')
    parser.add_argument('--dst_ckpt_dir',
                        default="/home/ma-user/work/mindformers/research/output/ckpt",
                        type=str,
                        help='path where to save dst ckpt')
    parser.add_argument('--prefix',
                        default='checkpoint_',
                        type=str,
                        help='prefix of transformed checkpoint')
    args = parser.parse_args()

    src_ckpt_strategy = get_strategy(args.src_ckpt_strategy)
    dst_ckpt_strategy = get_strategy(args.dst_ckpt_strategy)
    src_ckpt_dir = args.src_ckpt_dir
    dst_ckpt_dir = args.dst_ckpt_dir
    prefix = args.prefix

    assert os.path.exists(args.src_ckpt_dir), f'{args.src_ckpt_dir} not found!'

    print(f"src_ckpt_strategy: {src_ckpt_strategy}")
    print(f"dst_ckpt_strategy: {dst_ckpt_strategy}")
    print(f"src_ckpt_dir: {src_ckpt_dir}")
    print(f"dst_ckpt_dir: {dst_ckpt_dir}")
    print(f"prefix: {prefix}")

    print("......Start transform......")
    ms.transform_checkpoints(src_ckpt_dir, dst_ckpt_dir, prefix, src_ckpt_strategy, dst_ckpt_strategy)
    print("......Transform succeed!......")
```

## 5.3 修改配置文件

```bash
vi /home/ma-user/work/mindformers/research/baichuan2/run_baichuan2_7b_910b.yaml
```

修改权重初始化类型，将默认的 ”float16” 改为 ”float32”，保障推理精度(非必须，精度要求高场 景下可开启)

```yaml
seed: 0
output_dir: './output' # path to save checkpoint/strategy
load_checkpoint: 'baichuan2/7b/pytorch_model.ckpt'
src_strategy_path_or_dir: ''
auto_trans_ckpt: False  # If true, auto transform load_checkpoint to load in distributed model
only_save_strategy: False
resume_training: False
use_parallel: False
run_mode: 'predict'

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
    vocab_file: '/home/ma-user/work/mindformers/research/baichuan2/7b/tokenizer.model'
  type: LlamaProcessor
```

## 5.4 运行推理脚本

```bash
# 转换后的初始权重
cd /home/ma-user/work/mindformers/research/baichuan2/7b/

python /home/ma-user/work/mindformers/research/baichuan2/run_baichuan2.py \
--config /home/ma-user/work/mindformers/research/baichuan2/run_baichuan2_7b_910b.yaml \
--run_mode predict \
--use_parallel True \
--load_checkpoint /home/ma-user/work/mindformers/research/baichuan2/7b/output/transformed_checkpoint/7b/ \
--auto_trans_ckpt False \
--predict_data "<reserved_106>你是谁？<reserved_107>"

# 多卡推理
cd /home/ma-user/work/mindformers/research/

bash ./run_singlenode.sh \
"python /home/ma-user/work/mindformers/research/baichuan2/run_baichuan2.py \
--config /home/ma-user/work/mindformers/research/baichuan2/run_baichuan2_7b_910b.yaml \
--run_mode predict \
--use_parallel True \
--load_checkpoint /home/ma-user/work/mindformers/research/baichuan2/7b/output/transformed_checkpoint/7b/ \
--auto_trans_ckpt False \
--predict_data <reserved_106>你是谁？<reserved_107>" /user/config/jobstart_hccl.json [0,4] 4

# 预训练权重
cd /home/ma-user/work/mindformers/research/baichuan2/7b/

python /home/ma-user/work/mindformers/research/baichuan2/run_baichuan2.py \
--config /home/ma-user/work/mindformers/research/baichuan2/run_baichuan2_7b_910b.yaml \
--run_mode predict \
--use_parallel False \
--load_checkpoint /home/ma-user/work/mindformers/research/baichuan2/7b/rank_0/Baichuan2_7B_Chat.ckpt \
--auto_trans_ckpt False \
--predict_data "<reserved_106>什么是智慧园区?<reserved_107>"

# 微调过后的权重
cd /home/ma-user/work/mindformers/research/baichuan2/7b/

python /home/ma-user/work/mindformers/research/baichuan2/run_baichuan2.py \
--config /home/ma-user/work/mindformers/research/baichuan2/run_baichuan2_7b_910b.yaml \
--run_mode predict \
--use_parallel False \
--load_checkpoint /home/ma-user/work/mindformers/research/baichuan2/7b/output/merged_ckpt/rank_0/checkpoint_0.ckpt \
--auto_trans_ckpt False \
--predict_data "<reserved_106>帮助我制定一份去杭州的旅游攻略<reserved_107>"

```

如果报错: AttributeError: Baichuan2Tokenizer: 'tokenizers.AddedToken' object has no attribute 'special'

```bash
pip install tokenizers==0.15.0

```