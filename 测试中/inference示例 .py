import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'HangZhou_Ascend/Qwen2-0.5-Sd-v0.2.1'
device = "npu:0" # the device to load the model onto
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="npu"
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)

prompt = "Running boys"
messages = [
    {"role": "system", "content": "You are a helpful assistant to generate prompt for LLM model inputs."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)