"""
过滤权重参数
"""

import os
from glob import glob
import mindspore as ms

ignore_keys = [
    "accu_grads",
    "scale_sense",
    "global_step",
    "adam",
    "current_iterator_step",
    "last_overflow_iterator_step",
    "epoch_num",
    "step_num",
    "loss_scale",
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


if __name__ == "__main__":

    ckpt_path_or_dir = "/home/ma-user/work/mindformers/research/baichuan2/7b/output/checkpoint_network"
    assert os.path.exists(ckpt_path_or_dir), f"{ckpt_path_or_dir} not exists!"
    if os.path.isfile(ckpt_path_or_dir):
        ckpt_paths = [ckpt_path_or_dir]
    elif os.path.isdir(ckpt_path_or_dir):
        ckpt_paths = glob(os.path.join(ckpt_path_or_dir, "rank*/*.ckpt"))

    save_root = "/home/ma-user/work/mindformers/research/baichuan2/7b/output/filter_out"
    for ckpt_path in ckpt_paths:
        replace_part = ckpt_path.split("/rank")[0]
        save_path = ckpt_path.replace(replace_part, save_root)
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        only_save_model_param(ckpt_path, save_path)
