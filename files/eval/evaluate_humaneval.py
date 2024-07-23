import argparse
import tqdm
import jsonlines

import eval_utils
eval_utils.prepend_git_root_dir_to_python_path()
    
import mindspore as ms
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindformers.trainer.utils import set_seed


"""
git clone https://github.com/openai/human-eval
$ pip install -e human-eval

cd human-eval/data && zcat HumanEval.jsonl.gz > HumanEval.jsonl
cd ../..

vi human-eval/human_eval/execution.py  # uncomment line 58

pip install jsonlines

python evaluate_humaneval.py -f human-eval/data/HumanEval.jsonl

evaluate_functional_correctness HumanEval_res.jsonl
"""


def decode(tokens_list, tokenizer, raw_text_len):
    sents = []
    # print(len(tokens_list))
    for tokens in tokens_list:
        tokens = tokens.tolist()
        sent = tokenizer.tokenizer.decode(tokens[raw_text_len:])
        sent = sent.split("<|endoftext|>")[0]
        sent = sent.split("\n\n\n")[0]
        sent = sent.split("\n\n")[0]
        sent = sent.split("def ")[0]
        sents.append(sent)
    return sents


def generate_sample(model, tokenizer, input_txt):
    input_ids = tokenizer.tokenizer.encode(input_txt)
    raw_text_len = len(input_ids)
    context_enc = input_ids
    print(f"Input text: {input_txt}\n")
    outputs = model.generate(context_enc)
    output_text = decode(outputs, tokenizer, raw_text_len)[0]
    print(f"\nOutput text: \n{output_text}\n")
    return output_text


def main(args):
    model, tokenizer = eval_utils.load_model_and_tokenizer(args, use_past=True)

    f_output = jsonlines.Writer(open(args.sample_output_file, "w", encoding="utf-8"))

    f = jsonlines.open(args.sample_input_file)
    with f_output as output:
        for jobj in tqdm.tqdm(f, desc="task_idx"):
            prompt = jobj["prompt"]
            task_id = jobj["task_id"]
            gen_sents = generate_sample(model, tokenizer, prompt)
            gen_jobjs = {"task_id": task_id, "completion": gen_sents}
            output.write(gen_jobjs)
    f_output.close()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen model against OpenAI HumanEval evaluation set.")

    eval_utils.add_argparse_common_args(parser)

    group = parser.add_argument_group(title="Evaluation options")
    parser.add_argument(
        "-f",
        "--sample-input-file",
        type=str,
        required=True,
        help="data path to HumanEval.jsonl",
    )
    parser.add_argument(
        "-o", "--sample-output-file", type=str, default="HumanEval_res.jsonl"
    )

    args = parser.parse_args()

    main(args)
