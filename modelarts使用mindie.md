[当前文档访问路径](https://ai-fae.readthedocs.io/zh-cn/latest/modelarts使用mindie.html)

# 1.环境准备

硬件要求: 910B + 驱动24+

## 1.1 使用镜像
```
swr.cn-east-292.mygaoxinai.com/cloud/cann8.0.t63_python3.10.15_torch2.1.0_910b:20250208
```

## 1.2 安装MindIE
```bash
cd /home/ma-user/packages
# 下载torch_npu安装包
wget -O torch_npu-2.1.0.post8-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl https://gitee.com/ascend/pytorch/releases/download/v6.0.rc3-pytorch2.1.0/torch_npu-2.1.0.post8-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

cd /home/ma-user/work
# 获取安装脚本
wget http://39.171.244.84:30011/package_dependencies/script/mindie_new.sh
# 使脚本可执行
chmod +x mindie_new.sh
# 一键安装
bash mindie_new.sh
```

## 1.3 设置代理
```bash
echo 'export no_proxy=*.cn-east-292.mygaoxinai.com,*.cn-east-292.myhuaweicloud.com,pip.modelarts.private.com,localhost,127.0.0.1,192.168.1.1' >> ~/.bashrc

source ~/.bashrc
```

## 1.4 安装缺失依赖
```bash
pip install pydantic
pip install pytz
pip install gradio
```

# 2.下载模型

使用`魔搭社区`下载模型, 也可以使用`Huggingface`或`魔乐社区`

```bash
pip install modelscope

cd /home/ma-user/work
modelscope download --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --local_dir ./DeepSeek-R1-Distill-Qwen-32B

```

# 3.修改配置

修改模型配置，在下载模型权重的`config.json`文件中进行修改(参考中数据类型`"float16"/"bfloat16"`)

修改`MindIE-service`配置，MindIE工作目录位于 `/home/ma-user/Ascend/mindie/latest/mindie-service` 下

```bash
cd /home/ma-user/Ascend/mindie/latest/mindie-service
vi conf/config.json
```

```json
{
    "Version" : "1.0.0",
    "LogConfig" :
    {
        "logLevel" : "Info",
        "logFileSize" : 20,
        "logFileNum" : 20,
        "logPath" : "logs/mindservice.log"
    },

    "ServerConfig" :
    {
        "ipAddress" : "127.0.0.1",
        "managementIpAddress" : "127.0.0.2",
        "port" : 1025,
        "managementPort" : 1026,
        "metricsPort" : 1027,
        "allowAllZeroIpListening" : false,
        "maxLinkNum" : 1000,
        "httpsEnabled" : false, // 关闭https
        "fullTextEnabled" : false,
        "tlsCaPath" : "security/ca/",
        "tlsCaFile" : ["ca.pem"],
        "tlsCert" : "security/certs/server.pem",
        "tlsPk" : "security/keys/server.key.pem",
        "tlsPkPwd" : "security/pass/key_pwd.txt",
        "tlsCrl" : "security/certs/server_crl.pem",
        "managementTlsCaFile" : ["management_ca.pem"],
        "managementTlsCert" : "security/certs/management/server.pem",
        "managementTlsPk" : "security/keys/management/server.key.pem",
        "managementTlsPkPwd" : "security/pass/management/key_pwd.txt",
        "managementTlsCrl" : "security/certs/management/server_crl.pem",
        "kmcKsfMaster" : "tools/pmt/master/ksfa",
        "kmcKsfStandby" : "tools/pmt/standby/ksfb",
        "inferMode" : "standard",
        "interCommTLSEnabled" : false,
        "interCommPort" : 1121,
        "interCommTlsCaFile" : "security/grpc/ca/ca.pem",
        "interCommTlsCert" : "security/grpc/certs/server.pem",
        "interCommPk" : "security/grpc/keys/server.key.pem",
        "interCommPkPwd" : "security/grpc/pass/key_pwd.txt",
        "interCommTlsCrl" : "security/certs/server_crl.pem",
        "openAiSupport" : "vllm"
    },

    "BackendConfig" : {
        "backendName" : "mindieservice_llm_engine",
        "modelInstanceNumber" : 1,
        "npuDeviceIds" : [[0,1,2,3]], // 启用几卡推理，8卡则修改为[[0,1,2,3,4,5,6,7]]
        "tokenizerProcessNumber" : 8,
        "multiNodesInferEnabled" : false,
        "multiNodesInferPort" : 1120,
        "interNodeTLSEnabled" : true,
        "interNodeTlsCaFile" : "security/grpc/ca/ca.pem",
        "interNodeTlsCert" : "security/grpc/certs/server.pem",
        "interNodeTlsPk" : "security/grpc/keys/server.key.pem",
        "interNodeTlsPkPwd" : "security/grpc/pass/mindie_server_key_pwd.txt",
        "interNodeTlsCrl" : "security/grpc/certs/server_crl.pem",
        "interNodeKmcKsfMaster" : "tools/pmt/master/ksfa",
        "interNodeKmcKsfStandby" : "tools/pmt/standby/ksfb",
        "ModelDeployConfig" :
        {
            "maxSeqLen" : 2560,
            "maxInputTokenLen" : 2048,
            "truncation" : false,
            "ModelConfig" : [
                {
                    "modelInstanceType" : "Standard",
                    "modelName" : "deepseekr1", // 模型名称
                    "modelWeightPath" : "/home/ma-user/work/DeepSeek-R1-Distill-Qwen-32B", // 模型权重所在路径
                    "worldSize" : 4, // 加载卡的数量，指定8卡推理则修改为8
                    "cpuMemSize" : 5,
                    "npuMemSize" : -1,
                    "backendType" : "atb"
                }
            ]
        },

        "ScheduleConfig" :
        {
            "templateType" : "Standard",
            "templateName" : "Standard_LLM",
            "cacheBlockSize" : 128,

            "maxPrefillBatchSize" : 50,
            "maxPrefillTokens" : 8192,
            "prefillTimeMsPerReq" : 150,
            "prefillPolicyType" : 0,

            "decodeTimeMsPerReq" : 50,
            "decodePolicyType" : 0,

            "maxBatchSize" : 200,
            "maxIterTimes" : 512,
            "maxPreemptCount" : 0,
            "supportSelectBatch" : false,
            "maxQueueDelayMicroseconds" : 5000
        }
    }
}
```

完整配置文件参考:
https://www.hiascend.com/document/detail/zh/mindie/10RC3/mindieservice/servicedev/mindie_service0285.html

# 4.启动MindIE

```bash
cd /home/ma-user/Ascend/mindie/latest/mindie-service
./bin/mindieservice_daemon

# 控制台显示`Daemon start success!`则说明启动成功

# 如果启动失败, 可以执行下面两行命令开启debug, 然后执行启动mindie, 根据日志做对应修改直至启动成功
export MINDIE_LOG_LEVEL=debug
export MINDIE_LOG_TO_STDOUT=1

# 开启debug调试启动成功后, 先停了服务, 执行下面两行命令关闭debug, 然后执行启动mindie
unset MINDIE_LOG_LEVEL
unset MINDIE_LOG_TO_STDOUT

```

# 5.http请求测试
另起一个Terminal

```bash
curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{
  "inputs": "你是谁？",
  "parameters": {
    "best_of": 1,
    "decoder_input_details": false,
    "details": false,
    "do_sample": true,
    "max_new_tokens": 500,
    "repetition_penalty": 1.03,
    "return_full_text": true,
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

```

# 6.Gradio内网穿透
```bash
# 获取脚本
cd /home/ma-user/work
wget https://gitee.com/csw-assdasd8sa8d7as78/aicc_-docs/raw/master/source/part3/Gradio_Chat.py

vi Gradio_Chat.py
```
```py
    payload = {
        "model": "DeepSeek-R1-Distill-Qwen-32B", # 修改model为mindie配置中对应的模型名字
        "max_tokens": 2048,
        "messages": messages,
        "max_tokens": max_tokens,
        "presence_penalty": 1.03,
        "frequency_penalty": 1.0,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "stream": False
    }
```
```bash
get -O /home/ma-user/python3/lib/python3.10/site-packages/gradio/frpc_linux_arm64_v0.3 https://gitee.com/lincenying/public/raw/master/frpc_linux_arm64_v0.3

chmod +x /home/ma-user/python3/lib/python3.10/site-packages/gradio/frpc_linux_arm64_v0.3

python Gradio_Chat.py
# 执行脚本文件，会返回一个公共链接
# 浏览器输入返回的公共链接，即可打开Gradio界面
```

# 7.性能精度测试

## 7.1 数据集
```bash
mkdir -p /home/ma-user/work/data
cd /home/ma-user/work/data
# GSM8K
wget https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/test.jsonl
# CEval
wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
# MMLU
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
# BoolQ
wget https://github.com/svinkapeppa/boolq/blob/master/dev.jsonl
# HumanEval
wget https://github.com/openai/human-eval/blob/master/data/HumanEval.jsonl.gz

# 支持精度测试的数据集: GSM8K、CEval和MMLU
```

## 7.2 安装依赖
```bash
pip install jsonlines pyarrow prettytable 
```

## 7.3 性能测试
```bash
benchmark \
--DatasetPath "/home/ma-user/work/data/GSM8K" \
--DatasetType "gsm8k" \
--ModelName "DeepSeek-R1-Distill-Qwen-32B" \
--ModelPath "/home/ma-user/work/DeepSeek-R1-Distill-Qwen-32B" \
--TestType client \
--Concurrency 100 \
--Http http://127.0.0.1:1025 \
--MaxOutputLen 512

# --DatasetType 数据集类型，枚举值：ceval、gsm8k、boolq、humaneval和mmlu。
# --Concurrency 并发数，限制同时发起的连接数
```

性能测试结果主要关注FirstTokenTime、DecodeTime等token生成时延的指标和lpct（latency per compelete token，prefill阶段平均每个token时延）、Throughput等测试吞吐量的指标。

- `FirstTokenTime：首个token时延`
- `DecodeTime：Decode阶段时延`
- `Icpt：首token总时延/输入总token数。单位（ms）`
- `Throughput：整体测试过程的每秒请求数，吞吐量指标。单位（req/s）`

## 7.4 精度测试
```bash
benchmark \
--DatasetPath "/{数据集路径}/GSM8K" \
--DatasetType "gsm8k" \
--ModelName "Qwen1.5-7B" \
--ModelPath "/{权重路径}/Qwen1.5-7B" \
--TestType client \
--Concurrency 100 \
--Http http://127.0.0.1:1025 \
--MaxOutputLen 512 \
--TestAccuracy True

# --TestAccuracy True 参数是开启精度测试的开关
# 返回的accuracy字段为精度测试结果

```

## 7.5 参数说明
输入参数:
https://www.hiascend.com/document/detail/zh/mindie/10RC3/mindieservice/servicedev/mindie_service0153.html

输出参数:
https://www.hiascend.com/document/detail/zh/mindie/10RC3/mindieservice/servicedev/mindie_service0154.html

# 8.停止服务
```bash
ps -ef |grep mindieservice |awk '{print $2}'|xargs kill -9

```