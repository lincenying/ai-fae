

vi /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
cd /usr/local/Ascend/mindie/latest/mindie-service
./bin/mindieservice_daemon

pip install prettytable pyarrow jsonlines


benchmark \
--DatasetPath "/usr/local/Ascend/MindIE-LLM/tests/modeltest/dataset/full/CEval" \
--DatasetType "ceval" \
--ModelName "llama_65b" \
--ModelPath "/home/ma-user/huangming/wang/qwen/qwen-14b" \
--TestType client \
--Http http://127.0.0.1:1025 \
--Concurrency 1 \
--MaxOutputLen 20 \
--TaskKind stream \
--Tokenizer True \
--TestAccuracy True

benchmark \
--DatasetPath "/usr/local/Ascend/MindIE-LLM/tests/modeltest/dataset/full/CEval" \
--DatasetType "ceval" \
--ModelName "llama_65b" \
--ModelPath "/home/ma-user/huangming/wang/baichuan-inc/Baichuan2-13B" \
--TestType client \
--Http http://127.0.0.1:1025  \
--Concurrency 1 \
--MaxOutputLen 20 \
--TaskKind stream \
--Tokenizer True \
--TestAccuracy True


benchmark \
--DatasetPath "/usr/local/Ascend/MindIE-LLM/tests/modeltest/dataset/full/CEval" \
--DatasetType "ceval" \
--ModelName "baichuan2_13b" \
--ModelPath "/home/ma-user/huangming/wang/qwen/Qwen-14B" \
--TestType client \
--Http http://127.0.0.1/1025  \
--Concurrency 1 \
--MaxOutputLen 20 \
--TaskKind stream \
--Tokenizer True \
--TestAccuracy True

# qwen
curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{
  "inputs": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>请介绍三线圈电磁步进式设计。<|im_end|>\n<|im_start|>assistant",
  "parameters": {
    "best_of": 1,
    "decoder_input_details": true,
    "details": true,
    "do_sample": true,
    "max_new_tokens": 300,
    "repetition_penalty": 1.03,
    "return_full_text": false,
    "seed": null,
    "stop": [ 
      "photographer"
    ],
    "temperature": 0.5,
    "top_k": 10,
    "top_n_tokens": 5,
    "top_p": 0.95,
    "truncate": null,
    "typical_p": 0.95,
    "watermark": true
  },  
  "stream": false}' http://127.0.0.1:1025/ 

# baichuan
  curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{
    "inputs": "You are a helpful assistant.<reserved_106>\n核电厂机组大修过程中一回路水位的管理可以分为哪几个阶段？<reserved_107>\n",
    "parameters": {
      "best_of": 1,
      "decoder_input_details": false,
      "details": false,
      "do_sample": true,
      "max_new_tokens": 300,
      "repetition_penalty": 1.03,
      "return_full_text": false,
      "seed": null,
      "stop": [
        "photographer"
      ],
      "temperature": 0.5,
      "top_k": 10,
      "top_p": 0.95,
      "truncate": null,
      "typical_p": 0.95,
      "watermark": true
    },
    "stream": false}' http://127.0.0.1/1025 
  
bash /home/ma-user/work/mindformers/research/run_singlenode.sh \
"python /home/ma-user/work/mindformers/research/qwen/run_qwen.py \
--config /home/ma-user/work/mindformers/research/qwen/run_qwen_14b_notebook.yaml \
--load_checkpoint /home/ma-user/ckpt/qw/ \
--use_parallel True \
--run_mode finetune \
--auto_trans_ckpt True \
--train_data /home/ma-user/work/data/qwenalpa/alpaca.mindrecord" \
/user/config/jobstart_hccl.json [0,8] 8