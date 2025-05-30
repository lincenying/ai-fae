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
ps -ef |grep mindie |awk '{print $2}'|xargs kill -9
```

```bash
cd /home/ma-user/work/
# 下载obsutil
wget https://obs-community.obs.cn-north-1.myhuaweicloud.com/obsutil/current/obsutil_linux_arm64.tar.gz
# 解压缩obsutil
tar -zxvf obsutil_linux_arm64.tar.gz
# 修改可执行文件
chmod +x ./obsutil_linux_arm64_5.7.3/obsutil
# 移动obsutil
mv ./obsutil_linux_arm64_5.7.3 ./obs_bin
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

# xiemingda账号

### deepseek
obs://deepseekv3/deepseekR1-a8w8/
obs://deepseekv3/deepseekR1-bf16
obs://deepseekv3/deepseekR1
obs://deepseekv3/deepseekr1-w4a16
obs://deepseekv3/deepseekv3-bf16
obs://deepseekv3/DeepSeek-V3-0324-w8a8

### deepseek蒸馏
obs://bigmodel/DeepSDeepSeek-R1-Distill/DeepSeek-R1-Distill-Qwen-14B
obs://bigmodel/DeepSDeepSeek-R1-Distill/DeepSeek-R1-Distill-Qwen-32B
obs://bigmodel/DeepSDeepSeek-R1-Distill/DeepSeek-R1-Distill-Llama-70B

### qwen2
obs://bigmodel/qwen2-72b-instruct
obs://bigmodel/Qwen2-Audio-7B-Instruct
obs://bigmodel/Qwen2-VL-72B-Instruct

### qwen2.5
obs://bigmodel/QwQ-32B
obs://bigmodel/Qwen2.5-14B
obs://bigmodel/Qwen2.5-14B-Instruct
obs://bigmodel/Qwen2.5-32B-Instruct
obs://bigmodel/Qwen2.5-72B-Instruct
obs://bigmodel/Qwen2.5-VL-3B-Instruct
obs://bigmodel/qwen2.5-vl-7b-instruct
obs://bigmodel/Qwen2.5-VL-72B-Instruct
obs://bigmodel/qwen2.5-omni

### qwen3
obs://bigmodel/Qwen3/Qwen3-235B-A22B
obs://bigmodel/Qwen3/Qwen3-32B
obs://bigmodel/Qwen3/Qwen3-30B-A3B
obs://bigmodel/Qwen3/Qwen3-14B
obs://bigmodel/Qwen3/Qwen3-8B

`docker`

obs://docker/mindie_2.0.T18.B010-800I-A2-py3.11-openeuler24.03-lts-aarch64.tar.gz
obs://docker/mindie_2.0.RC1-800I-A2-py311-openeuler24.03-lts.tar.gz
obs://docker/mindie_2.0.T17.B010-800I-A2-py3.11-openeuler24.03-lts-aarch64.tar.gz
obs://docker/mindie_2.0.T3.1-800I-A2-py311-openeuler24.03-lts.tar.gz
obs://docker/mindie_2.0.T3-800I-A2-py311-openeuler24.03-lts.tar.gz
obs://docker/mindie_2.0.RC1.B081-800I-A2-py3.11-openeuler24.03-lts-aarch64.tar.gz
obs://docker/mindie_2.0.T9.B020-800I-A2-py3.11-openeuler24.03-lts-aarch64.tar.gz

obs://docker/mindie2.0.rc1-310p-py311.tar.gz
obs://docker/mindie_2.0.RC1.B081-300I-Duo-py3.11-openeuler24.03-lts-aarch64.tar.gz

obs://docker/mindie_1.0.T71-800I-A2-py311-ubuntu22.04-arm64.tar.gz

obs://docker/910A-ascend_24.1.rc3-cann_8.0.t63-py_3.10-ubuntu_20.04-aarch64-mindie_1.0.T71.05.tar
obs://docker/910A-ascend_24.1.rc3-cann_8.1.T17-py_3.11-ubuntu_20.04-aarch64-mindie_1.0.0.tar

obs://docker/vllm-ascend-v0.8.4rc2.tar.gz
obs://docker/vllm-ascend-v0.8.4rc1-openeuler.tar
obs://docker/vllm-ascend-v0.7.3rc2.tar.gz