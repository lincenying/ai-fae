import argparse

import torch
from openmind_hub import snapshot_download
from openmind import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the LLM model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to model",
        default=None,
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.model_name_or_path:
        model_path = args.model_name_or_path
    else:
        model_path = snapshot_download(
            "HangZhou_Ascend/Orbita-v0.1",
            revision="main",
            resume_download=True,
            ignore_patterns=["*.h5", "*.ot", "*.msgpack"],
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

    prompt = "Q: What is the largest animal?\nA:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)
    generation_output = model.generate(input_ids=input_ids, max_new_tokens=32)

    print(tokenizer.decode(generation_output[0]))


if __name__ == "__main__":
    main()

# python openmind-base.py --model_name_or_path /home/huangming/models/openmind/Orbita-v0.1/
