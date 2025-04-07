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

# xiemingda账号

obs://deepseekv3/deepseekR1-a8w8/
obs://deepseekv3/deepseekR1-bf16
obs://deepseekv3/deepseekR1
obs://deepseekv3/deepseekr1-w4a16
obs://deepseekv3/deepseekv3-bf16

obs://deepseekv3/DeepSeek-R1-Distill-Llama-70B
obs://bigmodel/DeepSeek-R1-Distill-Qwen-32B
obs://bigmodel/DeepSeek-R1-Distill-Qwen-14B
obs://bigmodel/Qwen2.5-14B
obs://bigmodel/Qwen2.5-32B-Instruct
obs://bigmodel/qwen2-72b-instruct
