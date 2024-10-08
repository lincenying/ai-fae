[当前文档访问路径](https://ai-fae.readthedocs.io/zh-cn/latest/obs包含数据.html)

# 镜像
```bash
docker pull swr.cn-east-292.mygaoxinai.com/huqs/mindspore2.2.14-cann7.0.0beta1_py_3.9-euler_2.8.3_910:v3

docker pull swr.cn-east-292.mygaoxinai.com/huqs/mindspore2.2.14-cann7.0.0beta1_py_3.9-euler_2.8.3_910:v2_qwen1_5_72b

docker pull swr.cn-east-292.mygaoxinai.com/huqs/mindspore2.2.10-cann7.0.0beta1_py_3.9-euler_2.8.3_910b:v2

docker pull swr.cn-east-292.mygaoxinai.com/huqs/pytorch2.1.0_cann8.0.rc1.alpha002_py3.9_euler2.8.3_910b:v8


```

```bash
ps -ef |grep run_qwen1_5.py |awk '{print $2}'|xargs kill -9
ps -ef |grep MindSpore/bin/python |awk '{print $2}'|xargs kill -9
```

```bash
cd /home/ma-user/work/
# 下载obsutil
wget https://obs-community.obs.cn-north-1.myhuaweicloud.com/obsutil/current/obsutil_linux_arm64.tar.gz
# 解压缩obsutil
tar -zxvf obsutil_linux_arm64.tar.gz
# 修改可执行文件
chmod +x ./obsutil_linux_arm64_5.5.12/obsutil
# 移动obsutil
mv ./obsutil_linux_arm64_5.5.12 ./obs_bin
# 添加环境变量
export PATH=$PATH:/home/ma-user/work/obs_bin
# 登录
obsutil config -i=###替换成AK### -k=###替换成SK### -e=obs.cn-east-292.mygaoxinai.com

# 将obs上的文件下载到notebook
obsutil cp obs://xxxx/abc.zip ./
# 将notebook上的文件上传到obs
obsutil cp ./def.zip obs://xxxx/
# 同步文件夹, 如将notebook的 ./xyz 文件夹同步到obs上
obsutil sync ./xyz obs://xxxx/xyz
# 同步文件夹, 如将obs上的 xyz 文件夹同步到notebook上
obsutil sync obs://xxxx/xyz ./xyz
```

复制文件夹
```bash
# 将output文件夹拷贝到obs:/model-data/qianwen1.5/7b/chat/ 目录下, 会创建output文件夹
obsutil cp ./output/ obs://model-data/qianwen1.5/7b/chat/ -f -r # 得到: obs://model-data/qianwen1.5/7b/chat/output/
# 将output文件夹里的文件拷贝到obs:/model-data/qianwen1.5/7b/chat/目录下, 不会创建output文件夹
obsutil cp ./output/ obs://model-data/qianwen1.5/7b/chat/ -f -r -flat
obsutil cp ./output_baichuan2_910b/ obs://model-data/baichuan2/ -f -r # 得到: obs://model-data/baichuan2/output_baichuan2_910b/
obsutil cp ./qwen1_5-72b.ckpt obs://model-data/qianwen1.5/72b/ # 得到: obs://model-data/qianwen1.5/72b/qwen1_5-72b.ckpt
```

同步文件夹
```bash
obsutil sync ./glmv2 obs://huangming/mindformers/research/glmv2
obsutil sync ./qwen obs://huangming/mindformers/research/qwen
```

移动文件夹
```bash
# 将xxx文件夹拷贝到obs://model-data/qianwen1.5/7b/ 目录下, 会创建xxx文件夹
obsutil mv obs://model-data/qianwen1.5/7b/xxx/ obs://model-data/qianwen1.5/7b/ -f -r
# 将xxx文件夹内文件拷贝到obs://model-data/qianwen1.5/7b/ 目录下, 不会创建xxx文件夹
obsutil mv obs://model-data/qianwen1.5/7b/xxx/ obs://model-data/qianwen1.5/7b/ -f -r
obsutil mv obs://model-data/chatglm3/ obs://model-data/chatglm32k/ -f -r
obsutil mv obs://model-data/chatglm32k/chatglm3/dev.json obs://model-data/chatglm32k/ # ==> obs://model-data/chatglm32k/dev.json

```

obs内部移动文件
```bash
obsutil mv obs://model-data/qianwen1.5/14b/model-00001-of-00008.safetensors obs://model-data/qianwen1.5/14b/base/

```

# yukai账号

## baichuan2

### 7b-chat
```
obs://model-data/baichuan2/Baichuan2_7B_Chat.ckpt
obs://model-data/baichuan2/pytorch_model.bin
```

### 13b-base
```
obs://model-data/baichuan2/Baichuan2-13B-Base.ckpt
```

### 13b-chat
```
obs://model-data/baichuan2/Baichuan2-13B-Chat.ckpt
```

### 数据集
```
obs://model-data/baichuan2/belle_chat_ramdon_10k.json

obs://model-data/baichuan2/models # 910a权重及相关文件
obs://model-data/baichuan2/output # 910a微调后相关文件
```

## chatglm2

### 6b
```
obs://model-data/chatglm2/glm2_6b.ckpt
```

## chatglm3

### 6b
```
obs://model-data/chatgml3/glm3/6b/rank_0/glm3_6b.ckpt
```

## chatglm32k

### 6b
```
obs://model-data/chatglm32k/pytorch_model-00001-of-00007.bin
obs://model-data/chatglm32k/pytorch_model-00002-of-00007.bin
obs://model-data/chatglm32k/pytorch_model-00003-of-00007.bin
obs://model-data/chatglm32k/pytorch_model-00004-of-00007.bin
obs://model-data/chatglm32k/pytorch_model-00005-of-00007.bin
obs://model-data/chatglm32k/pytorch_model-00006-of-00007.bin
obs://model-data/chatglm32k/pytorch_model-00007-of-00007.bin
obs://model-data/chatglm32k/pytorch_model-00008-of-00007.bin
obs://model-data/chatglm32k/glm32k.ckpt
obs://model-data/chatglm32k/train.json
obs://model-data/chatglm32k/dev.json
obs://model-data/chatglm32k/data.zip
```

## qianwen

### 7b-base
```
# 原始权重
obs://model-data/qianwen/7b_base/
# 转换后权重
obs://model-data/qianwen/qwen_7b_base.ckpt
```

### 7b-instruct
```
obs://model-data/qianwen/7b-instruct/
```

### 7b-chat
```
obs://model-data/qianwen/7b_chat/model-00001-of-00008.safetensors
obs://model-data/qianwen/7b_chat/model-00002-of-00008.safetensors
obs://model-data/qianwen/7b_chat/model-00003-of-00008.safetensors
obs://model-data/qianwen/7b_chat/model-00004-of-00008.safetensors
obs://model-data/qianwen/7b_chat/model-00005-of-00008.safetensors
obs://model-data/qianwen/7b_chat/model-00006-of-00008.safetensors
obs://model-data/qianwen/7b_chat/model-00007-of-00008.safetensors
obs://model-data/qianwen/7b_chat/model-00008-of-00008.safetensors

```

### 14b-base
```
# 转换后权重
obs://model-data/qianwen/14b_base
# 原始权重
obs://model-data/qianwen/qwen_14b_base.ckpt
```

### 14-chat

原始权重
```
obs://model-data/qianwen/14b-chat/
```

转换后权重
```
obs://model-data/qianwen/qwen_14b_chat.ckpt
```

### 数据集
```
obs://model-data/qianwen/qwen.tiktoken
obs://model-data/qianwen/alpaca_data.json
```

## qianwen1.5

### 7b-base
```
#转换后权重
obs://model-data/qianwen1.5/7b/base/qwen15_7b_base.ckpt

#原始权重
obs://model-data/qianwen1.5/7b/base/model-00001-of-00004.safetensors
obs://model-data/qianwen1.5/7b/base/model-00002-of-00004.safetensors
obs://model-data/qianwen1.5/7b/base/model-00003-of-00004.safetensors
obs://model-data/qianwen1.5/7b/base/model-00004-of-00004.safetensors
```

### 7b-chat
```
#转换后权重
obs://model-data/qianwen1.5/7b/chat/qwen15_7b_chat.ckpt

#原始权重
obs://model-data/qianwen1.5/7b/chat/model-00001-of-00004.safetensors
obs://model-data/qianwen1.5/7b/chat/model-00002-of-00004.safetensors
obs://model-data/qianwen1.5/7b/chat/model-00003-of-00004.safetensors
obs://model-data/qianwen1.5/7b/chat/model-00004-of-00004.safetensors
```

### 14b-base
```
obs://model-data/qianwen1.5/14b/base/model-00001-of-00008.safetensors
obs://model-data/qianwen1.5/14b/base/model-00002-of-00008.safetensors
obs://model-data/qianwen1.5/14b/base/model-00003-of-00008.safetensors
obs://model-data/qianwen1.5/14b/base/model-00004-of-00008.safetensors
obs://model-data/qianwen1.5/14b/base/model-00005-of-00008.safetensors
obs://model-data/qianwen1.5/14b/base/model-00006-of-00008.safetensors
obs://model-data/qianwen1.5/14b/base/model-00007-of-00008.safetensors
obs://model-data/qianwen1.5/14b/base/model-00008-of-00008.safetensors
```

### 14b-chat
[见xiemingda账号](#qwen1_5-14b-chat_1)

### 72b-chat
[见xiemingda账号](#qwen1_5-72b-chat_1)


## llama2

### 7b
```
obs://model-data/llama2/llama2_7b.ckpt
```

## llama3

### 8B
[见xiemingda账号](#llama3-8b_1)

## yolov5

### 数据集
[见xiemingda账号](#yolov5-dataset_1)

# xiemingda账号

## llama2
```
obs://temp-zjw/llama2-7b/pytorch_model-00001-of-00002.bin
obs://temp-zjw/llama2-7b/pytorch_model-00002-of-00002.bin
```

## llama3

### llama3-8b_1
```
obs://llama3/data/Meta-Llama-3-8B-Instruct
```

## glm32k
```
obs://glm32k/glm3_6b.ckpt
obs://glm32k/data.zip
```

## yolov5

### yolov5-dataset_1
```
obs://temp-zjw/datasets/coco2017.zip
```

## qwen1.5

### qwen1_5-7b-chat_1
```
obs://wio/qwen1.5-7b-chat-ckpt/qwen15_7b_chat.ckpt
```

### qwen1_5-14b-chat_1
```
obs://wio/qwen1.5-14b-chat/model-00001-of-00008.safetensors
obs://wio/qwen1.5-14b-chat/model-00002-of-00008.safetensors
obs://wio/qwen1.5-14b-chat/model-00003-of-00008.safetensors
obs://wio/qwen1.5-14b-chat/model-00004-of-00008.safetensors
obs://wio/qwen1.5-14b-chat/model-00005-of-00008.safetensors
obs://wio/qwen1.5-14b-chat/model-00006-of-00008.safetensors
obs://wio/qwen1.5-14b-chat/model-00007-of-00008.safetensors
obs://wio/qwen1.5-14b-chat/model-00008-of-00008.safetensors
```

## qwen1_5-72b-chat_1
```
obs://wio/qw1.5-72b-chat.ckpt
```