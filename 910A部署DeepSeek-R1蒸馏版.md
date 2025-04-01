[当前文档访问路径](https://ai-fae.readthedocs.io/zh-cn/latest/910A部署DeepSeek-R1蒸馏版.html)

# 1. 环境准备
## 1.1 服务器要求

910A

# 2. 配置裸金属相关环境
## 2.1 安装驱动
使用`ssh`连接裸金属后, 执行以下命令:
```bash
mkdir -p /home/hm
yum update
# 解决 Do you want to try build driver after input kernel absolute path? 报错
yum install kernel-devel
yum install wget
wget http://39.171.244.84:30011/drivers/HDK%2024.1.RC3/Ascend-hdk-910-npu-driver_24.1.rc3_linux-aarch64.run
wget http://39.171.244.84:30011/drivers/HDK%2024.1.RC3/Ascend-hdk-910-npu-firmware_7.5.0.1.129.run
chmod +x Ascend-hdk-910-npu-driver_24.1.rc3_linux-aarch64.run Ascend-hdk-910-npu-firmware_7.5.0.1.129.run

# 更新驱动
./Ascend-hdk-910-npu-driver_24.1.rc3_linux-aarch64.run  --upgrade

# 安装驱动
./Ascend-hdk-910-npu-firmware_7.5.0.1.129.run --full

# 重启系统
reboot
```

```bash
# 安装docker
cd ~
vi docker.sh
```

写入以下内容

```
#!/bin/bash

#下载docker包
wget https://download.docker.com/linux/static/stable/aarch64/docker-28.0.4.tgz

#解压
tar zxf docker-28.0.4.tgz

#移动解压后的文件夹到/usr/bin
mv docker/* /usr/bin

#写入docker.service
cat >/usr/lib/systemd/system/docker.service <<EOF
[Unit]
Description=Docker Application Container Engine
Documentation=https://docs.docker.com
After=network-online.target firewalld.service
Wants=network-online.target
[Service]
Type=notify
ExecStart=/usr/bin/dockerd
ExecReload=/bin/kill -s HUP $MAINPID
LimitNOFILE=infinity
LimitNPROC=infinity
TimeoutStartSec=0
Delegate=yes
KillMode=process
Restart=on-failure
StartLimitBurst=3
StartLimitInterval=60s
[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload

#启动docker
systemctl start docker

#设置开机自启动
systemctl enable docker

#查看docker版本
docker version
```

```bash
bash ./docker.sh

# 安装docker-runtime
wget https://gitee.com/ascend/mind-cluster/releases/download/v6.0.0/Ascend-docker-runtime_6.0.0_linux-aarch64.run
chmod +x Ascend-docker-runtime_6.0.0_linux-aarch64.run
./Ascend-docker-runtime_6.0.0_linux-aarch64.run --install

systemctl daemon-reload && systemctl restart docker
```

# 3. 准备容器
## 3.1 准备容器

```bash
# 下载镜像
docker pull swr.cn-central-221.ovaijisuan.com/wh-aicc-fae/mindie:910A-ascend_24.1.rc3-cann_8.0.t63-py_3.10-ubuntu_20.04-aarch64-mindie_1.0.T71.05
```

## 3.2下载模型

使用`魔搭社区`下载模型, 也可以使用`Huggingface`或`魔乐社区`

```bash
pip install modelscope

cd /home/hm
modelscope download --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --local_dir ./DeepSeek-R1-Distill-Qwen-32B

```
或者使用`obsutil`下载

```bash
cd /home/hm/
# 下载obsutil
wget https://obs-community.obs.cn-north-1.myhuaweicloud.com/obsutil/current/obsutil_linux_arm64.tar.gz
# 解压缩obsutil
tar -zxvf obsutil_linux_arm64.tar.gz
# 修改可执行文件
chmod +x ./obsutil_linux_arm64_5.5.12/obsutil
# 移动obsutil
mv ./obsutil_linux_arm64_5.5.12 ./obs_bin
# 添加环境变量
echo 'export OBSAK="替换成AK"' >> ~/.bashrc
echo 'export OBSSK="替换成SK"' >> ~/.bashrc
echo 'export PATH=$PATH:/home/hm/obs_bin' >> ~/.bashrc
source ~/.bashrc

obsutil config -i=${OBSAK} -k=${OBSSK} -e=obs.cn-east-292.mygaoxinai.com

obsutil cp obs://bigmodel/DeepSeek-R1-Distill-Qwen-14B/ ./DeepSeek-R1-Distill-Qwen-14B/ -f -r -flat
obsutil cp obs://bigmodel/DeepSeek-R1-Distill-Qwen-32B/ ./DeepSeek-R1-Distill-Qwen-32B/ -f -r -flat
obsutil cp obs://deepseekv3/DeepSeek-R1-Distill-Llama-70B/ ./DeepSeek-R1-Distill-Llama-70B/ -f -r -flat

```

## 3.3 启动容器

使用`docker images`查看下载下来的`image`的ID，
使用下面启动命令(参考)：

```bash
docker run -itd --privileged  --name=mindie-server --net=host \
--shm-size 500g \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device /dev/devmm_svm \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /usr/local/sbin:/usr/local/sbin \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /home/hm:/home/hm \
swr.cn-central-221.ovaijisuan.com/wh-aicc-fae/mindie:910A-ascend_24.1.rc3-cann_8.0.t63-py_3.10-ubuntu_20.04-aarch64-mindie_1.0.T71.05

# 如果报 owner not right /usr/bin/runc 1000 错误, 执行:
chown root:root /usr/bin/runc
```

或者使用:

```bash
docker run -itd --privileged=true --name=mindie-server --net=host --ipc=host \
--shm-size 500g \
-e ASCEND_VISIBLE_DEVICES=0-7 \
-v /home/hm:/home/hm \
0d1f23321380
```

进入容器:
```bash
# docker ps 查看下容器ID
docker exec -it bdf31455d47a /bin/bash
```

# 4. MindIE服务化启动
## 4.1 修改模型配置文件

修改权重路径中config.json中的torch_dtype为float16

```bash
vi /home/hm/DeepSeek-R1-Distill-Qwen-32B/config.json
# 将 "torch_dtype": "bfloat16" 该成 "torch_dtype": "float16"
```

## 4.2 修改服务化参数

```bash
cd /usr/local/Ascend/mindie/latest/mindie-service
vi conf/config.json

```
修改以下参数:
```json
{
    "Version" : "1.0.0",
    "LogConfig" :
    {
        "logLevel" : "Info",
        "logFileSize" : 20,
        "logFileNum" : 20,
        "logPath" : "logs/mindie-server.log"
    },

    "ServerConfig" :
    {
        "ipAddress" : "192.168.0.24",
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
        "tlsCrlPath" : "security/certs/",
        "tlsCrlFiles" : ["server_crl.pem"],
        "managementTlsCaFile" : ["management_ca.pem"],
        "managementTlsCert" : "security/certs/management/server.pem",
        "managementTlsPk" : "security/keys/management/server.key.pem",
        "managementTlsPkPwd" : "security/pass/management/key_pwd.txt",
        "managementTlsCrlPath" : "security/management/certs/",
        "managementTlsCrlFiles" : ["server_crl.pem"],
        "kmcKsfMaster" : "tools/pmt/master/ksfa",
        "kmcKsfStandby" : "tools/pmt/standby/ksfb",
        "inferMode" : "standard",
        "interCommTLSEnabled" : true,
        "interCommPort" : 1121,
        "interCommTlsCaPath" : "security/grpc/ca/",
        "interCommTlsCaFiles" : ["ca.pem"],
        "interCommTlsCert" : "security/grpc/certs/server.pem",
        "interCommPk" : "security/grpc/keys/server.key.pem",
        "interCommPkPwd" : "security/grpc/pass/key_pwd.txt",
        "interCommTlsCrlPath" : "security/grpc/certs/",
        "interCommTlsCrlFiles" : ["server_crl.pem"],
        "openAiSupport" : "vllm"
    },

    "BackendConfig" : {
        "backendName" : "mindieservice_llm_engine",
        "modelInstanceNumber" : 1,
        "npuDeviceIds" : [[0,1,2,3,4,5,6,7]], // 启用几卡推理，8卡则修改为[[0,1,2,3,4,5,6,7]]
        "tokenizerProcessNumber" : 8,
        "multiNodesInferEnabled" : false,
        "multiNodesInferPort" : 1120,
        "interNodeTLSEnabled" : true,
        "interNodeTlsCaPath" : "security/grpc/ca/",
        "interNodeTlsCaFiles" : ["ca.pem"],
        "interNodeTlsCert" : "security/grpc/certs/server.pem",
        "interNodeTlsPk" : "security/grpc/keys/server.key.pem",
        "interNodeTlsPkPwd" : "security/grpc/pass/mindie_server_key_pwd.txt",
        "interNodeTlsCrlPath" : "security/grpc/certs/",
        "interNodeTlsCrlFiles" : ["server_crl.pem"],
        "interNodeKmcKsfMaster" : "tools/pmt/master/ksfa",
        "interNodeKmcKsfStandby" : "tools/pmt/standby/ksfb",
        "ModelDeployConfig" :
        {
            "maxSeqLen" : 25600, // 1
            "maxInputTokenLen" : 20480,
            "truncation" : true, // 2
            "ModelConfig" : [
                {
                    "modelInstanceType" : "Standard",
                    "modelName" : "DeepSeek-R1-Distill-Qwen-32B",
                    "modelWeightPath" : "/home/hm/DeepSeek-R1-Distill-Qwen-32B",
                    "worldSize" : 8,
                    "cpuMemSize" : 5,
                    "npuMemSize" : -1,
                    "backendType" : "atb",
                    "trustRemoteCode" : false
                }
            ]
        },

        "ScheduleConfig" :
        {
            "templateType" : "Standard",
            "templateName" : "Standard_LLM",
            "cacheBlockSize" : 128,

            "maxPrefillBatchSize" : 1, // 3
            "maxPrefillTokens" : 25600, // 4
            "prefillTimeMsPerReq" : 150,
            "prefillPolicyType" : 0,

            "decodeTimeMsPerReq" : 50,
            "decodePolicyType" : 0,

            "maxBatchSize" : 50, // 5
            "maxIterTimes" : 20480, // 6
            "maxPreemptCount" : 0,
            "supportSelectBatch" : false,
            "maxQueueDelayMicroseconds" : 5000
        }
    }
}
```

## 4.3 拉起服务
```bash
cd /usr/local/Ascend/mindie/latest/mindie-service
./bin/mindieservice_daemon

# 如果启动失败, 可以执行下面两行命令开启debug, 然后执行启动mindie
export MINDIE_LOG_LEVEL=debug
export MINDIE_LOG_TO_STDOUT=1

# 开启debug调试启动成功后, 先停了服务, 执行下面两行命令关闭debug, 然后执行启动mindie
unset MINDIE_LOG_LEVEL
unset MINDIE_LOG_TO_STDOUT

```

## 4.4 openai接口

另外新起一个窗口（也要进入docker），输入命令发送POST请求：
```bash
curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{
    "inputs": "你是谁",
    "parameters": {
        "max_new_tokens": 5012
    },
    "stream": false
}' http://192.168.0.24:1025/

curl -H "Accept: application/json" -H "Content-Type: application/json" -X POST -d '{
    "model": "DeepSeek-R1-Distill-Qwen-32B",
    "messages": [{
        "role": "user",
        "content": "介绍下杭州西湖"
    }],
    "max_tokens": 512,
    "stream": false
}' http://127.0.0.1:1025/v1/chat/completions

```

## 4.5 结束mindie服务
```bash
ps -ef |grep mindie |awk '{print $2}'|xargs kill -9
```