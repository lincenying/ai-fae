[当前文档访问路径](https://ai-fae.readthedocs.io/zh-cn/latest/LoRA_ckpt权重转换为safetensors权重.html)

# lora - ckpt转safetensors流程

## 1. 获取权重
参考训练流程获取权重文件

## 2. 合并权重
使用脚本合并权重，因为是lora微调的权重，需要使用merged_lora.py脚本
```py
# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""transform lora_ckpt"""
import os
import argparse
from collections import OrderedDict
import sys
sys.path.insert(0,"/home/ma-user/work/mindformers/")
import mindspore as ms
from mindspore import Parameter, Tensor
import mindspore.ops as P
from mindformers.tools.logger import logger

def get_strategy(startegy_path, rank_id=None):
    """Merge strategy if strategy path is dir

    Args:
        startegy_path (str): The path of stategy.
        rank_id (int): The rank id of device.

    Returns:
        None or strategy path
    """
    if not startegy_path or startegy_path == "None":
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

def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_ckpt_strategy',
                        default="",
                        help='path of src ckpt strategy')
    parser.add_argument('--dst_ckpt_strategy',
                        default="",
                        help='path of dst ckpt strategy')
    parser.add_argument('--src_ckpt_path_or_dir',
                        default="",
                        type=str,
                        help='path of src ckpt')
    parser.add_argument('--dst_ckpt_dir',
                        default="",
                        type=str,
                        help='path where to save dst ckpt')
    parser.add_argument('--prefix',
                        default='checkpoint_',
                        type=str,
                        help='prefix of transformed checkpoint')
    parser.add_argument('--lora_scaling',
                        default=0.25,
                        type=float,
                        help='scale of lora when merge model weight, default is lora_alpha/lora_rank')
    args = parser.parse_args()

    src_ckpt_strategy = get_strategy(args.src_ckpt_strategy)
    dst_ckpt_strategy = get_strategy(args.dst_ckpt_strategy)
    src_ckpt_path_or_dir = args.src_ckpt_path_or_dir
    dst_ckpt_dir = args.dst_ckpt_dir
    prefix = args.prefix
    lora_scaling = args.lora_scaling

    logger.info(f"src_ckpt_strategy: {src_ckpt_strategy}")
    logger.info(f"dst_ckpt_strategy: {dst_ckpt_strategy}")
    logger.info(f"src_ckpt_path_or_dir: {src_ckpt_path_or_dir}")
    logger.info(f"dst_ckpt_dir: {dst_ckpt_dir}")
    logger.info(f"prefix: {prefix}")

    if not os.path.isdir(src_ckpt_path_or_dir):
        logger.info("......Only Need MergeLora......")
        src_lora_ckpt_path = src_ckpt_path_or_dir
    else:
        logger.info("......Need Merge&Trans......")
        logger.info("......Start Transckpt......")
        ms.transform_checkpoints(src_ckpt_path_or_dir, dst_ckpt_dir, prefix, src_ckpt_strategy, dst_ckpt_strategy)
        logger.info("......Complete Trans&Save......")
        src_lora_ckpt_path = dst_ckpt_dir + "/rank_0/" + prefix + "0.ckpt"
        logger.info("src_lora_ckpt_path---------------")
        logger.info(src_lora_ckpt_path)
    logger.info("......Start Merge Lorackpt......")
    param_dict = ms.load_checkpoint(src_lora_ckpt_path)
    lora_keys = [k for k in param_dict if 'lora_a' in k]
    non_lora_keys = [k for k in param_dict if not 'lora_' in k]
    param_dict_lora = OrderedDict()
    for k in non_lora_keys:
        param_dict_lora[k] = param_dict[k].clone()
    for k in lora_keys:
        if k.split('.')[0] in ['adam_m', 'adam_v']:
            continue
        logger.info(f'Merging {k}')
        original_key = k.replace('_lora_a', '').replace('mindpet_delta', 'weight')
        assert original_key in param_dict
        lora_a_key = k
        lora_b_key = k.replace('lora_a', 'lora_b')
        original_value = param_dict_lora[original_key]
        param_dict_lora[original_key] = Parameter(Tensor(P.add(original_value, P.mm(param_dict[lora_b_key], \
                                         param_dict[lora_a_key]) * lora_scaling), original_value.dtype), \
                                         name=original_key)
    logger.info("......Start save merged ckpt......")
    save_checkpoint_file_name = os.path.join(dst_ckpt_dir, 'merged_lora.ckpt')
    ms.save_checkpoint(param_dict_lora, save_checkpoint_file_name)
    logger.info("......Merge succeed!.......")
```
src权重使用 训练输出路径output/checkpoint_network，src策略文件使用 训练输出路径output/strategy/ckpt_strategy_rank_0_rank_0.ckpt，dest目录指定一个新目录 ./merged_ckpt

## 3. 转换为bin权重

```bash
python convert_reversed.py --./merged_ckpt/merged_lora.ckpt --torch_ckpt_path ./bin/qwen1_5_7b_fp16.bin
```

convert_reversed.py

```py
# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Convert llama weight.
Support mindspore format and Meta format.
"""

import json
import argparse
import torch
import mindspore as ms


def pt2ms(value: torch.Tensor, dtype) -> ms.Tensor:
    """
    convert torch.Tensor to ms.Tensor with specified dtype
    """
    if value.dtype == torch.bfloat16:
        np_value = value.to(torch.float32).numpy()
    else:
        np_value = value.detach().numpy()

    if dtype:
        return ms.Tensor(np_value, dtype=dtype)
    return ms.Tensor(np_value, dtype=ms.bfloat16) if value.dtype == torch.bfloat16 else ms.Tensor(np_value)


def ms2pt(value: ms.Tensor, dtype) -> torch.Tensor:
    """
    convert ms.Tensor to torch.Tensor with specified dtype
    """
    if value.dtype == ms.bfloat16:
        np_value = value.data.astype(ms.float32).asnumpy()
    else:
        np_value = value.data.asnumpy()

    if dtype:
        return torch.from_numpy(np_value).to(dtype)
    return torch.from_numpy(np_value).to(torch.bfloat16) if value.dtype == ms.bfloat16 else torch.from_numpy(np_value)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def name_replace(name: str):
    """replace ms param name to hf."""
    name = name.replace('tok_embeddings.embedding_weight', 'embed_tokens.weight')
    name = name.replace('.attention.wq.', '.self_attn.q_proj.')
    name = name.replace('.attention.wk.', '.self_attn.k_proj.')
    name = name.replace('.attention.wv.', '.self_attn.v_proj.')
    name = name.replace('.attention.wo.', '.self_attn.o_proj.')
    name = name.replace('.feed_forward.w1.', '.mlp.gate_proj.')
    name = name.replace('.feed_forward.w2.', '.mlp.down_proj.')
    name = name.replace('.feed_forward.w3.', '.mlp.up_proj.')
    name = name.replace('.attention_norm.', '.input_layernorm.')
    name = name.replace('.ffn_norm.', '.post_attention_layernorm.')
    name = name.replace('.norm_out.', '.norm.')
    return name

# pylint: disable=W0613
def convert_ms_to_pt(input_path, output_path, dtype=None, **kwargs):
    """convert ms weight to hf."""
    print(f"Trying to convert mindspore checkpoint in '{input_path}'.", flush=True)
    model_ms = ms.load_checkpoint(input_path)

    state_dict = {}
    for name, value in model_ms.items():
        name = name_replace(name)
        print(f'\rprocessing parameter: {name} {value.shape}     ', end='', flush=True)
        state_dict[name] = ms2pt(value, dtype)

    torch.save(state_dict, output_path)
    print(f"\rConvert mindspore checkpoint finished, the huggingface checkpoint is saved in '{output_path}'.",
          flush=True)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mindspore_ckpt_path', default='transform.ckpt')
    parser.add_argument('--torch_ckpt_path', default='./qwen2/qwen2-hf/')
    args = parser.parse_args()
    convert_ms_to_pt(input_path=args.mindspore_ckpt_path, output_path=args.torch_ckpt_path)

```

## 4. 转换为safetensors
```bash
python bin2st.py --model_path  ./bin/qwen1_5_7b_fp16.bin
```

模型输出在bin文件所在目录下
bin2st.py

```py
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import argparse
import os

from atb_llm.utils.convert import convert_files
from atb_llm.utils.hub import weight_files
from atb_llm.utils.log import logger


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help="model and tokenizer path",
                        default='/data/acltransformer_testdata/weights/llama2/llama-2-70b',
                        )
    return parser.parse_args()


def convert_bin2st(model_path):
    local_pt_files = weight_files(model_path, revision=None, extension=".bin")
    local_st_files = [
        p.parent / f"{p.stem.lstrip('pytorch_')}.safetensors"
        for p in local_pt_files
    ]
    convert_files(local_pt_files, local_st_files, discard_names=[])
    found_st_files = weight_files(model_path)


def convert_bin2st_from_pretrained(model_path):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        low_cpu_mem_usage=True,
        torch_dtype="auto")
    model.save_pretrained(model_path, safe_serialization=True)


if __name__ == '__main__':
    args = parse_arguments()

    try:
        convert_bin2st(args.model_path)
    except RuntimeError:
        logger.warning('convert weights failed with torch.load method, need model loaded to convert')
        convert_bin2st_from_pretrained(args.model_path)
```
