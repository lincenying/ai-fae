import argparse

import torch
from openmind_hub import snapshot_download
from openmind import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to model files",
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
            "HangZhou_Ascend/nox-solar-10.7b-v4nox-solar-10.7b-v4",
            revision="main",
            resume_download=True,
            ignore_patterns=["*.h5", "*.ot", "*.msgpack"],
        )

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = model.eval()
    inputs = tokenizer(["Come to the beautiful nature"], return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to(model.device)
    gen_kwargs = {"max_length": 128, "top_p": 0.8, "temperature": 0.8, "do_sample": True, "repetition_penalty": 1.0}
    output = model.generate(**inputs, **gen_kwargs)
    output = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    print(output)


if __name__ == "__main__":
    main()


# python openmind-chat.py --model_name_or_path /home/huangming/models/openmind/nox-solar-10.7b-v4/
