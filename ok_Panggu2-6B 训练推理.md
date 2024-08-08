[当前文档访问路径](https://ai-fae.readthedocs.io/zh-cn/latest/ok_Panggu2-6B%20训练推理.html)

# 1.镜像

swr.cn-east-292.mygaoxinai.com/huqs/mindspore2.2.10-cann7.0.0beta1_py_3.9-euler_2.8.3_910b:v2

# 2. 安装mindformer
```bash
git clone -b r1.0 https://gitee.com/mindspore/mindformers.git
cd mindformers
bash build.sh
```
# 3. 数据集准备(参考llama 2文档)
wikitext 数据 ：https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/dataset/wikitext-2/wikitext-2-v1.zip

词表
https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/pangualpha/vocab.model

转数据mind
```bash
python pretrain_data_process.py \
--dataset_type wiki \
--input_glob /home/ma-user/work/mindformers/dataset/wikitext-2/wiki.train.tokens \
--model_file /home/ma-user/work/mindformers/dataset/vocab.model \
--seq_length 1025 \
--output_file /home/ma-user/work/mindformers/dataset/wiki1025.mindrecord
```
# 4.训练
配置
![图片](assets/IMG_10.png)
```bash
cd scripts
bash run_distribute.sh /user/config/nbstart_hccl.json ../configs/pangualpha/run_pangualpha_2_6b.yaml [0,8] finetune 8
```
# 5. 推理
脚本 单卡推理 mindformer
```bash
python run_mindformer.py --config configs/pangualpha/run_pangualpha_2_6b.yaml --run_mode predict --predict_data 上联：欢天喜地度佳节 下联： --use_parallel False
# output result is: [{'text_generation_text': ['上联:欢天喜地度佳节 下联:笑逐颜开迎佳期 横批:幸福快乐<eot>']}]
```