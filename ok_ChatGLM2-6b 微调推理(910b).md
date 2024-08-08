[当前文档访问路径](https://ai-fae.readthedocs.io/zh-cn/latest/ok_ChatGLM2-6b%20微调推理(910b).html)

- 镜像: mindspore2.2.10-cann7.0.0beta1_py_3.9-euler_2.8.3_910b:v2
- 镜像源: swr.cn-east-292.mygaoxinai.com/huqs/mindspore2.2.10-cann7.0.0beta1_py_3.9-euler_2.8.3_910b:v2
- Ascend: 4*ascend-d910b|CPU: 96核 768GB

## 1. 安装 Mindformers

```bash
git clone -b r1.0 https://gitee.com/mindspore/mindformers.git # 建议选这个版本
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

```bash
mdkdir -p /home/ma-user/work/mindformers/research/glmv2
cd /home/ma-user/work/mindformers/research/glmv2
wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/glm2/glm2_6b.ckpt
wget https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/glm2/tokenizer.model

```

# 3. 数据准备
 
 ```bash
wget https://cloud.tsinghua.edu.cn/seafhttp/files/02997df9-d20d-4b97-9b88-7f4256e42921/AdvertiseGen.tar.gz
tar -zxvf AdvertiseGen.tar.gz

```

# 4. 模型训练

## 4.1 修改配置文件

```bash
vi /home/ma-user/work/mindformers/configs/glm2/run_glm2_6b_finetune_910b.yaml
```

```yaml
load_checkpoint: '/home/ma-user/work/mindformers/research/glmv2/glm2_6b.ckpt'
train_dataset: &train_dataset
  data_loader:
    dataset_dir: "/home/ma-user/work/mindformers/research/glmv2/AdvertiseGen/train.json"
  tokenizer:
    vocab_file: "/home/ma-user/work/mindformers/research/glmv2/tokenizer.model"

eval_dataset: &eval_dataset
  data_loader:
    dataset_dir: "/home/ma-user/work/mindformers/research/glmv2/AdvertiseGen/dev.json"
  tokenizer:
    vocab_file: "/home/ma-user/work/mindformers/research/glmv2/tokenizer.model"

# 根据多卡情况修改
parallel_config:
  data_parallel: 4
  model_parallel: 1
  pipeline_stage: 1
  expert_parallel: 1
  micro_batch_num: 1
```

## 4.2 启动微调训练脚本

```bash
cd /home/ma-user/work/mindformers/scripts

bash run_distribute.sh /user/config/jobstart_hccl.json ../configs/glm2/run_glm2_6b_finetune_910b.yaml '[0,4]' finetune

```

## 5. 推理

## 5.1 过滤权重参数

```bash
cd /home/ma-user/work/mindformers/output
vi filter_ckpt_param.py
```

```py
import os
from glob import glob
import mindspore as ms

ignore_keys = [
  'accu_grads',
  'scale_sense',
  'global_step',
  'adam',
  'current_iterator_step',
  'last_overflow_iterator_step',
  'epoch_num',
  'step_num',
  'loss_scale'
]

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
    
    ckpt_path_or_dir = '/home/ma-user/work/mindformers/output/checkpoint_network'
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
                        default="/home/ma-user/work/mindformers/output/strategy",
                        help='path of src ckpt strategy')
    parser.add_argument('--dst_ckpt_strategy',
                        default="",
                        help='path of dst ckpt strategy')
    parser.add_argument('--src_ckpt_dir',
                        default="/home/ma-user/work/mindformers/output/filter_out",
                        type=str,
                        help='path of src ckpt')
    parser.add_argument('--dst_ckpt_dir',
                        default="/home/ma-user/work/mindformers/output/ckpt",
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

```bash
python ./transform_ckpt.py
```

## 5.3 推理

```bash
cd /home/ma-user/work/mindformers/
vi infer.py
```

```py
from mindformers import AutoConfig, AutoModel, AutoTokenizer, ChatGLM2Tokenizer
import mindspore as ms

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)

# **注意** LoRA微调模型替换成 “glm2_6b_lora”,
# **注意** P-Tuning 微调模型替换成 “glm2_6b_ptuning2”
config = AutoConfig.from_pretrained("glm2_6b")
# 可以在此使用下行代码指定自定义权重进行推理，默认使用自动从obs上下载的预训练权重
#
# 以下两个权重, 根据情况2选1
#
# 原始权重
config.checkpoint_name_or_path = "/home/ma-user/work/mindformers/research/glmv2/glm2_6b.ckpt"
# 微调过后权重
config.checkpoint_name_or_path = "/home/ma-user/work/mindformers/output/ckpt/rank_0/checkpoint_0.ckpt"
#
config.use_past = True
config.seq_length = 1024
model = AutoModel.from_config(config)

# 以下两种tokenizer实例化方式选其一即可
# 1. 在线加载方式
#tokenizer = AutoTokenizer.from_pretrained("glm2_6b")
# 2. 本地加载方式
tokenizer = ChatGLM2Tokenizer("/home/ma-user/work/mindformers/research/glmv2/tokenizer.model")

kwargs={}
gen_kwargs = {"max_length": config.seq_length, "num_beams": 1, "do_sample": False, "top_p": 3,"top_k": 0.7,
              "temperature": 1, **kwargs}

queries = ["你好", "请介绍一下杭州", "那里有什么好吃的吗"]
history = []
for query in queries:
    # 如果想关闭history，此处传入 `history=[]` 即可
    prompt = tokenizer.build_prompt(query, history=history)
    input_id = tokenizer(prompt)["input_ids"]

    output = model.generate([input_id], **gen_kwargs)

    # output 包括了[input_id, output]两个部分
    output = output[0][len(input_id):]
    response = tokenizer.decode(output)
    print(response)
    history += [(query, response)]
```

```bash
python infer.py
```