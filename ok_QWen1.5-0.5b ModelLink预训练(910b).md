- 镜像: `swr.cn-east-292.mygaoxinai.com/cloud/pytorch2.1.0-cann8.0.rc2-python3.9-910b:202407260923`  
- 硬件：Ascend: 8*ascend-d910b|CPU: 192核 1536GB

# 依赖安装
```
pip install numpy==1.22.4
pip install datasets
```
# 数据集
## 分词器
```
cd /home/ma-user/work/ModelLink/
mkdir ./model_from_hf/qwen15-0.5b-hf/
cd ./model_from_hf/qwen15-0.5b-hf/
wget https://hf-mirror.com/Qwen/Qwen1.5-0.5B/resolve/main/config.json
wget https://hf-mirror.com/Qwen/Qwen1.5-0.5B/resolve/main/generation_config.json
wget https://hf-mirror.com/Qwen/Qwen1.5-0.5B/resolve/main/merges.txt
wget https://hf-mirror.com/Qwen/Qwen1.5-0.5B/resolve/main/tokenizer.json
wget https://hf-mirror.com/Qwen/Qwen1.5-0.5B/resolve/main/tokenizer_config.json
wget https://hf-mirror.com/Qwen/Qwen1.5-0.5B/resolve/main/vocab.json
cd ../../
```
### 下载分词器备选方案 （文件夹已经在上一步创建）
```
/home/ma-user/work/obsutil/obsutil config -i=GY7OH6UQBYGBDNZEXJFO -k=lTpaHt14Uw5c854reLbCRGbQ9XYlZgc6ayAnl6ls -e=obs.cn-east-292.mygaoxinai.com
/home/ma-user/work/obsutil/obsutil cp -r -f obs://model-data/0726/model/qwen15-0.5b-hf/ /home/ma-user/work/ModelLink/model_from_hf/
cd /home/ma-user/work/ModelLink/
```

## 数据集
```
cd /home/ma-user/work/ModelLink/dataset
wget https://hf-mirror.com/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..
```
### 下载数据集备选方案
```
/home/ma-user/work/obsutil/obsutil config -i=GY7OH6UQBYGBDNZEXJFO -k=lTpaHt14Uw5c854reLbCRGbQ9XYlZgc6ayAnl6ls -e=obs.cn-east-292.mygaoxinai.com
/home/ma-user/work/obsutil/obsutil cp obs://model-data/0726/dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet /home/ma-user/work/ModelLink/dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet
```
## 处理数据   
```
cd /home/ma-user/work/ModelLink/
mkdir ./dataset/qwen15-0.5b-hf/
python ./tools/preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/qwen15-0.5b-hf/ \
    --output-prefix ./dataset/qwen15-0.5b-hf/alpaca \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF
```

# 预训练
## 修改脚本
```
vim /home/ma-user/work/ModelLink/examples/qwen15/pretrain_qwen15_0point5b_ptd.sh
```
```
源代码
# please fill these path configurations
CKPT_LOAD_DIR="your model ckpt path"
CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
替换新代码
CKPT_SAVE_DIR="./ckpt/qwen15-0.5b-hf/"
TOKENIZER_PATH="./model_from_hf/qwen15-0.5b-hf"
DATA_PATH="./dataset/qwen15-0.5b-hf/alpaca_text_document"

源代码
TP=1
PP=1
替换新代码
TP=8
PP=1

源代码
GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 24 \
    --hidden-size 1024 \
    --ffn-hidden-size 2816 \
    --num-attention-heads 16 \
    --load ${CKPT_LOAD_DIR} \
删除此处的    --load ${CKPT_LOAD_DIR} \
```

# 开始预训练
```
cd /home/ma-user/work/ModelLink/
bash examples/qwen15/pretrain_qwen15_0point5b_ptd.sh
```