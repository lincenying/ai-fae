import os
import pandas as pd
import numpy as np
import argparse
from collections import defaultdict

from tqdm import tqdm

import eval_utils
eval_utils.prepend_git_root_dir_to_python_path()

import mindspore as ms
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype

from mindformers.trainer.utils import set_seed

"""
wget https://huggingface.co/datasets/haonan-li/cmmlu/resolve/main/cmmlu_v1_0_1.zip
mkdir -p data/cmmlu
cd data/cmmlu; unzip ../../cmmlu_v1_0_1.zip
cd ../../
python evaluate_cmmlu.py -d data/cmmlu/
"""


def format_example(line, include_answer=True):
    example = "问题：" + line["Question"]
    for choice in choices:
        example += f'\n{choice}. {line[f"{choice}"]}'

    if include_answer:
        example += "\n答案：" + line["Answer"] + "\n\n"
    else:
        example += "\n答案："
    return example


def generate_few_shot_prompt(k, subject, dev_df):
    prompt = ""
    if k == -1:
        k = dev_df.shape[0]
    for i in range(k):
        prompt += format_example(
            dev_df.iloc[i, :],
            include_answer=True,
        )
    return prompt


def get_logits(tokenizer, model, prompt: str):
    input_len = len(tokenizer.tokenizer.encode(prompt))
    input_ids = tokenizer([prompt], padding="max_length", truncation=True)["input_ids"]
    input_ids = Tensor(input_ids)
    tokens = {"input_ids": input_ids}

    # FIXME: what if input_len > model.seq_length?
    outputs, _, _ = model(input_ids)  # logits, input_ids, input_mask
    logits = outputs[:, input_len - 1, :]
    log_probs = ms.ops.softmax(logits, axis=-1)
    return log_probs, {"tokens": tokens}


def eval_subject(
    model,
    tokenizer,
    subject_name,
    test_df,
    k=5,
    dev_df=None,
    few_shot=False,
    save_result_dir=None,
    **kwargs,
):
    result = []
    score = []

    few_shot_prompt = (
        generate_few_shot_prompt(k, subject_name, dev_df) if few_shot else []
    )
    all_probs = {"prob_A": [], "prob_B": [], "prob_C": [], "prob_D": []}
    if args.debug:
        print(f"few_shot_prompt: {few_shot_prompt}")

    for _, row in tqdm(test_df.iterrows(),
                       total=len(test_df),
                       desc=subject_name.rjust(30)):
        question = format_example(row, include_answer=False)
        full_prompt = few_shot_prompt + question

        output, input_info = get_logits(tokenizer, model, full_prompt)
        assert output.shape[0] == 1
        logits = output.flatten()

        softval = ms.ops.softmax(
            Tensor(
                [
                    logits[tokenizer("A")["input_ids"]].numpy(),
                    logits[tokenizer("B")["input_ids"]].numpy(),
                    logits[tokenizer("C")["input_ids"]].numpy(),
                    logits[tokenizer("D")["input_ids"]].numpy(),
                ]
            ),
            axis=0,
        )
        if softval.dtype in {mstype.float16}:
            softval = softval.to(dtype=mstype.float32)
        probs = softval.numpy()

        for i, choice in enumerate(choices):
            all_probs[f"prob_{choice}"].append(probs[i])
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        if "Answer" in row:
            correct = 1 if pred == row["Answer"] else 0
            score.append(correct)
            if args.debug:
                print(f'{question} pred: {pred} ref: {row["Answer"]}')
        result.append(pred)

    if score:
        correct_ratio = 100 * sum(score) / len(score)
        if args.debug:
            print(subject_name, correct_ratio)
    else:
        correct_ratio = 0
    if save_result_dir:
        test_df["model_output"] = result
        for i, choice in enumerate(choices):
            test_df[f"prob_{choice}"] = all_probs[f"prob_{choice}"]
        if score:
            test_df["correctness"] = score
        os.makedirs(save_result_dir, exist_ok=True)
        test_df.to_csv(
            os.path.join(save_result_dir, f"{subject_name}_result.csv"),
            encoding="utf-8",
            index=False,
        )

    return correct_ratio


def cal_cmmlu(res):
    print("\n\n\n")
    res = {k.split("-")[-1]: float(v) for k, v in res.items()}
    for k, v in TASK_NAME_MAPPING.items():
        avg_acc = np.mean(list(map(lambda x: res[x], v)))
        print(f"{k} acc: {avg_acc:.2f}")
    avg_all_acc = np.mean(list(res.values()))
    print(f"AVERAGE acc: {avg_all_acc:.2f}")


subcategories = {
    "agronomy": ["other"],
    "anatomy": ["biology"],
    "ancient_chinese": ["linguistics", "china specific"],
    "arts": ["arts"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "chinese_civil_service_exam": ["politics", "china specific"],
    "chinese_driving_rule": ["other", "china specific"],
    "chinese_food_culture": ["culture", "china specific"],
    "chinese_foreign_policy": ["politics", "china specific"],
    "chinese_history": ["history", "china specific"],
    "chinese_literature": ["literature", "china specific"],
    "chinese_teacher_qualification": ["education", "china specific"],
    "college_actuarial_science": ["math"],
    "college_education": ["education"],
    "college_engineering_hydrology": ["engineering"],
    "college_law": ["law"],
    "college_mathematics": ["math"],
    "college_medical_statistics": ["statistics"],
    "clinical_knowledge": ["other"],
    "college_medicine": ["other"],
    "computer_science": ["computer science"],
    "computer_security": ["other"],
    "conceptual_physics": ["physics"],
    "construction_project_management": ["other", "china specific"],
    "economics": ["economics"],
    "education": ["education"],
    "elementary_chinese": ["linguistics", "china specific"],
    "elementary_commonsense": ["other", "china specific"],
    "elementary_information_and_technology": ["other"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "ethnology": ["culture", "china specific"],
    "food_science": ["other"],
    "genetics": ["biology"],
    "global_facts": ["global"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_geography": ["geography"],
    "high_school_mathematics": ["math"],
    "high_school_physics": ["physics"],
    "high_school_politics": ["politics", "china specific"],
    "human_sexuality": ["other"],
    "international_law": ["law"],
    "journalism": ["sociology"],
    "jurisprudence": ["law"],
    "legal_and_moral_basis": ["other"],
    "logical": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "marxist_theory": ["philosophy"],
    "modern_chinese": ["linguistics", "china specific"],
    "nutrition": ["other"],
    "philosophy": ["philosophy"],
    "professional_accounting": ["business"],
    "professional_law": ["law"],
    "professional_medicine": ["other"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_study": ["politics"],
    "sociology": ["culture"],
    "sports_science": ["other"],
    "traditional_chinese_medicine": ["other", "china specific"],
    "virology": ["biology"],
    "world_history": ["history"],
    "world_religions": ["global"],
}

categories = {
    "STEM": [
        "physics",
        "chemistry",
        "biology",
        "computer science",
        "math",
        "engineering",
        "statistics",
    ],
    "Humanities": ["history", "philosophy", "law", "arts", "literature", "global"],
    "Social Science": [
        "linguistics",
        "business",
        "politics",
        "culture",
        "economics",
        "geography",
        "psychology",
        "education",
        "sociology",
    ],
    "Other": ["other"],
    "China specific": ["china specific"],
}

TASK_NAME_MAPPING = defaultdict(list)
for k, v in categories.items():
    for subject, subcat in subcategories.items():
        for c in subcat:
            if c in v:
                TASK_NAME_MAPPING[k].append(subject)


choices = ["A", "B", "C", "D"]


def main(args):
    model, tokenizer = eval_utils.load_model_and_tokenizer(args, use_past=False)

    test_result = {}
    for subject_name in tqdm(subcategories.keys(), desc='TOTAL'.ljust(30)):
        dev_file_path = os.path.join(args.eval_data_path, "dev", f"{subject_name}.csv")
        test_file_path = os.path.join(
            args.eval_data_path, "test", f"{subject_name}.csv"
        )
        dev_df = pd.read_csv(dev_file_path)
        test_df = pd.read_csv(test_file_path)

        score = eval_subject(
            model,
            tokenizer,
            subject_name,
            dev_df=dev_df,
            test_df=test_df,
            k=5,
            few_shot=True,
            save_result_dir=f"outs/cmmlu_eval_result",
        )
        test_result[subject_name] = score
    cal_cmmlu(test_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation Qwen model against CMMLU dataset.")

    eval_utils.add_argparse_common_args(parser)

    # Provide extra arguments required for tasks
    group = parser.add_argument_group(title="Evaluation options")
    group.add_argument(
        "-d", "--eval_data_path", type=str, required=True, help="Path to eval data"
    )
    
    args = parser.parse_args()
    set_seed(args.seed)

    main(args)
