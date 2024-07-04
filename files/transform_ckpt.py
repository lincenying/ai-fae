"""
合并权重
"""

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

    assert os.path.exists(startegy_path), f"{startegy_path} not found!"

    if os.path.isfile(startegy_path):
        return startegy_path

    if os.path.isdir(startegy_path):
        if rank_id:
            merge_path = os.path.join(startegy_path, f"merged_ckpt_strategy_{rank_id}.ckpt")
        else:
            merge_path = os.path.join(startegy_path, f"merged_ckpt_strategy.ckpt")

        if os.path.exists(merge_path):
            os.remove(merge_path)

        ms.merge_pipeline_strategys(startegy_path, merge_path)
        return merge_path

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_ckpt_strategy", default="/home/ma-user/work/mindformers/research/baichuan2/7b/output/strategy", help="path of src ckpt strategy"
    )
    parser.add_argument("--dst_ckpt_strategy", default="", help="path of dst ckpt strategy")
    parser.add_argument(
        "--src_ckpt_dir",
        default="/home/ma-user/work/mindformers/research/baichuan2/7b/output/filter_out",
        type=str,
        help="path of src ckpt",
    )
    parser.add_argument(
        "--dst_ckpt_dir",
        default="/home/ma-user/work/mindformers/research/baichuan2/7b/output/merged_ckpt",
        type=str,
        help="path where to save dst ckpt",
    )
    parser.add_argument("--prefix", default="checkpoint_", type=str, help="prefix of transformed checkpoint")
    args = parser.parse_args()

    src_ckpt_strategy = get_strategy(args.src_ckpt_strategy)
    dst_ckpt_strategy = get_strategy(args.dst_ckpt_strategy)
    src_ckpt_dir = args.src_ckpt_dir
    dst_ckpt_dir = args.dst_ckpt_dir
    prefix = args.prefix

    assert os.path.exists(args.src_ckpt_dir), f"{args.src_ckpt_dir} not found!"

    print(f"src_ckpt_strategy: {src_ckpt_strategy}")
    print(f"dst_ckpt_strategy: {dst_ckpt_strategy}")
    print(f"src_ckpt_dir: {src_ckpt_dir}")
    print(f"dst_ckpt_dir: {dst_ckpt_dir}")
    print(f"prefix: {prefix}")

    print("......Start transform......")
    ms.transform_checkpoints(src_ckpt_dir, dst_ckpt_dir, prefix, src_ckpt_strategy, dst_ckpt_strategy)
    print("......Transform succeed!......")
