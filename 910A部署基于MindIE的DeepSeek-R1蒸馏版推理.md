[当前文档访问路径](https://ai-fae.readthedocs.io/zh-cn/latest/910A部署基于MindIE的DeepSeek-R1蒸馏版推理.html)

# 1. 环境准备
## 1.1 服务器要求

910A

# 2. 配置裸金属相关环境
## 2.1 安装驱动
使用`ssh`连接裸金属后, 执行以下命令:
```bash
mkdir -p /data

npu-smi info # 如果驱动版本大等于24.1.rc3, 以下安装更新驱动步骤可省略

# 如果操作系统是Ubuntu 以下步骤可省略 ↓↓↓↓
yum update -y
# 解决 Do you want to try build driver after input kernel absolute path? 报错
yum install kernel-devel -y
yum install wget -y
# 如果操作系统是Ubuntu 以上步骤可省略 ↑↑↑↑

mkdir -p /data/drivers
cd /data/drivers

# 官网下载地址: https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=10&cann=8.0.0.beta1&driver=Ascend+HDK+24.1.0
wget http://39.171.244.84:30011/drivers/HDK%2024.1.RC3/Ascend-hdk-910-npu-driver_24.1.rc3_linux-aarch64.run
wget http://39.171.244.84:30011/drivers/HDK%2024.1.RC3/Ascend-hdk-910-npu-firmware_7.5.0.1.129.run
chmod +x Ascend-hdk-910-npu-driver_24.1.rc3_linux-aarch64.run Ascend-hdk-910-npu-firmware_7.5.0.1.129.run

# 更新驱动
./Ascend-hdk-910-npu-driver_24.1.rc3_linux-aarch64.run  --upgrade

# 如果安装kernel-devel还报Do you want to try build driver after input kernel absolute path?错误, 则输两次y, 然后输入下面路径
# /lib/modules/4.19.36-vhulk1907.1.0.h1665.eulerosv2r8.aarch64/build

# 安装驱动
./Ascend-hdk-910-npu-firmware_7.5.0.1.129.run --full

```
## 2.2 安装docker

### 2.2.1 方法1, 使用yum安装
```bash
yum install docker-ce docker-ce-cli containerd.io

```
### 2.2.2 (推荐)方法2, 通过二进制包手动安装
```bash
cd ~
cat > docker.sh << 'EOF'
#!/bin/bash

# 下载docker包
# 根据架构适当修改, 详情见: https://mirrors.aliyun.com/docker-ce/linux/static/stable/
wget https://mirrors.aliyun.com/docker-ce/linux/static/stable/aarch64/docker-28.0.4.tgz

# 解压
tar zxf docker-28.0.4.tgz

# 移动解压后的文件夹到/usr/bin
mv docker/* /usr/bin

# 写入docker.service开始 ===>
cat > /usr/lib/systemd/system/docker.service << EOF_DOCKER_SERVICE
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
EOF_DOCKER_SERVICE

# <=== 写入docker.service结束

# 重新加载服务
systemctl daemon-reload

# 启动docker
systemctl start docker

# 设置开机自启动
systemctl enable docker

# 查看docker版本
docker version

EOF

# 执行docker.sh
bash ./docker.sh

```

```bash
# 安装docker-runtime
wget https://gitee.com/ascend/mind-cluster/releases/download/v6.0.0/Ascend-docker-runtime_6.0.0_linux-aarch64.run
chmod +x Ascend-docker-runtime_6.0.0_linux-aarch64.run
./Ascend-docker-runtime_6.0.0_linux-aarch64.run --install

systemctl daemon-reload && systemctl restart docker

```

## 2.3 安装 obsutil(可选)

```bash
cd /data
# 下载obsutil
wget https://obs-community.obs.cn-north-1.myhuaweicloud.com/obsutil/current/obsutil_linux_arm64.tar.gz
# 解压缩obsutil
tar -zxvf obsutil_linux_arm64.tar.gz
# 修改可执行文件
chmod +x ./obsutil_linux_arm64_5.7.3/obsutil
# 移动obsutil
mv ./obsutil_linux_arm64_5.7.3 ./obs_bin
# 添加环境变量
echo 'export OBSAK="替换成AK"' >> ~/.bashrc
echo 'export OBSSK="替换成SK"' >> ~/.bashrc
echo 'export PATH=$PATH:/data/obs_bin' >> ~/.bashrc
source ~/.bashrc

obsutil config -i=${OBSAK} -k=${OBSSK} -e=obs.cn-east-292.mygaoxinai.com

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

cd /data
# 根据情况下载所需要模型
modelscope download --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --local_dir ./DeepSeek-R1-Distill-Qwen-32B
modelscope download --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --local_dir ./DeepSeek-R1-Distill-Qwen-7B
modelscope download --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --local_dir ./DeepSeek-R1-Distill-Qwen-14B
modelscope download --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B --local_dir ./DeepSeek-R1-Distill-Llama-70B

```
或者使用`obsutil`下载, 需执行步骤 2.3

```bash
cd /data

# 根据情况下载所需要模型
obsutil cp obs://bigmodel/DeepSeek-R1-Distill/DeepSeek-R1-Distill-Qwen-32B/ ./DeepSeek-R1-Distill-Qwen-32B/ -f -r -flat
obsutil cp obs://bigmodel/DeepSeek-R1-Distill/DeepSeek-R1-Distill-Qwen-14B ./DeepSeek-R1-Distill-Qwen-14B/ -f -r -flat
obsutil cp obs://bigmodel/DeepSeek-R1-Distill/DeepSeek-R1-Distill-Llama-70B/ ./DeepSeek-R1-Distill-Llama-70B/ -f -r -flat

```

## 3.3 启动容器

使用`docker images`查看下载下来的`image`的ID，
使用下面启动命令(参考)：

```bash
# 如果报 owner not right /usr/bin/runc 1000 错误, 执行:
chown root:root /usr/bin/runc

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
-v /data:/data \
swr.cn-central-221.ovaijisuan.com/wh-aicc-fae/mindie:910A-ascend_24.1.rc3-cann_8.0.t63-py_3.10-ubuntu_20.04-aarch64-mindie_1.0.T71.05

```

或者使用:

```bash
docker run -itd --privileged=true --name=mindie-server --net=host --ipc=host \
--shm-size 500g \
-e ASCEND_VISIBLE_DEVICES=0-7 \
-v /data:/data \
0d1f23321380
```

进入容器:
```bash
# docker ps 查看下容器ID 或者使用 容器name
docker exec -it mindie-server /bin/bash
```

# 4. MindIE服务化启动
## 4.1 修改模型配置文件

修改权重路径中config.json中的torch_dtype为float16

```bash
vi /data/DeepSeek-R1-Distill-Qwen-32B/config.json
# 将 "torch_dtype": "bfloat16" 该成 "torch_dtype": "float16"
```

## 4.2 修改服务化参数

```bash
cd /usr/local/Ascend/mindie/latest/mindie-service
vi conf/config.json

```
修改以下带注释的参数:
```json
{
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
    },

    "BackendConfig" : {
        "npuDeviceIds" : [[0,1,2,3]], // 启用几卡推理，910A推荐4卡
        "ModelDeployConfig" :
        {
            "maxSeqLen" : 25600, // 1
            "maxInputTokenLen" : 20480,
            "truncation" : true, // 2
            "ModelConfig" : [
                {
                    "modelInstanceType" : "Standard",
                    "modelName" : "DeepSeek-R1-Distill-Qwen-32B",
                    "modelWeightPath" : "/data/DeepSeek-R1-Distill-Qwen-32B",
                    "worldSize" : 4,
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

或者通过命令行快速修改

```bash
# 替换IP (192.168.0.23 替换成 本机对应的IP)
sed -i 's/"ipAddress"[[:space:]]*:[[:space:]]*".*",/"ipAddress" : "192.168.0.23",/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 替换端口 (2025 替换成需要监听的端口)
sed -i 's/"port"[[:space:]]*:[[:space:]]*.*,/"port" : 2025,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 关闭https (false 为关闭, true为开启)
sed -i 's/"httpsEnabled"[[:space:]]*:[[:space:]]*.*,/"httpsEnabled" : false,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 启用几卡 ([[0,1,2,3,4,5,6,7]] 为8卡, [[0,1,2,3]] 则为4卡, 根据需要修改)
sed -i 's/"npuDeviceIds"[[:space:]]*:[[:space:]]*\[\[.*\]\],/"npuDeviceIds" : [[0,1,2,3,4,5,6,7]],/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 替换模型名称 (这个命名影响服务启动, 但是推理时模型名称需要和这个对应)
sed -i 's/"modelName"[[:space:]]*:[[:space:]]*".*",/"modelName" : "DeepSeek-R1-Distill-Qwen-32B",/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 替换模型路径 (模型权重的绝对路径)
sed -i 's/"modelWeightPath"[[:space:]]*:[[:space:]]*".*",/"modelWeightPath" : "\/home\/hm\/DeepSeek-R1-Distill-Qwen-32B\/",/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 启用几卡 (4 代表4卡, 需要和上面的 npuDeviceIds 保持一致)
sed -i 's/"worldSize"[[:space:]]*:[[:space:]]*.*,/"worldSize" : 4,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 其他修改
sed -i 's/"maxPrefillBatchSize"[[:space:]]*:[[:space:]]*.*,/"maxPrefillBatchSize" : 1,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
sed -i 's/"maxSeqLen"[[:space:]]*:[[:space:]]*.*,/"maxSeqLen" : 25600,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
sed -i 's/"maxPrefillTokens"[[:space:]]*:[[:space:]]*.*,/"maxPrefillTokens" : 25600,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
sed -i 's/"maxBatchSize"[[:space:]]*:[[:space:]]*.*,/"maxBatchSize" : 50,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
sed -i 's/"maxIterTimes"[[:space:]]*:[[:space:]]*.*,/"maxIterTimes" : 20480,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 查看结果
cat /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json | grep -E 'ipAddress|port|httpsEnabled|multiNodesInferEnabled|interCommTLSEnabled|interNodeTLSEnabled|npuDeviceIds|modelName|modelWeightPath|worldSize|maxPrefillBatchSize|maxSeqLen|maxPrefillTokens|maxInputTokenLen|maxBatchSize|maxIterTimes'

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


# 5. 服务化常见问题
若出现out of memory报错，可适当调高NPU_MEMORY_FRACTION环境变量（默认值为0.8），适当调低服务化配置文件config.json中maxSeqLen、maxInputTokenLen、maxPrefillBatchSize、maxPrefillTokens、maxBatchSize等参数。

```bash
export NPU_MEMORY_FRACTION=0.96
```
 
若出现hccl通信超时报错，可配置以下环境变量。

```bash
export HCCL_CONNECT_TIMEOUT=7200 # 该环境变量需要配置为整数，取值范围[120,7200]，单位s
export HCCL_EXEC_TIMEOUT=0
```
 
无进程内存残留
如果卡上有内存残留，且有进程，可以尝试以下指令：

```bash
pkill -9 -f 'mindie|python'
```

如果卡上有内存残留，但无进程，可以尝试以下指令：

```bash
npu-smi set -t reset -i 0 -c 0 #重启npu卡
npu-smi info -t health -i <card_idx> -c 0 #查询npu告警
```

例：

```bash
npu-smi set -t reset -i 0 -c 0 #重启npu卡0
npu-smi info -t health -i 2 -c 0 #查询npu卡2告警
```

如果卡上有进程残留，无进程，且重启NPU卡无法消除残留内存，请尝试reboot重启机器

日志收集
遇到推理报错时，请打开日志环境变量，收集日志信息。

算子库日志|默认输出路径为"~/atb/log"

```bash
export ASDOPS_LOG_LEVEL = INFO
export ASDOPS_LOG_TO_FILE = 1
```

加速库日志|默认输出路径为"~/mindie/log/debug"

```bash
export ATB_LOG_LEVEL = INFO
export ATB_LOG_TO_FILE = 1
```

MindIE Service日志|默认输出路径为"~/mindie/log/debug"

```bash
export MINDIE_LOG_TO_FILE = 1
export MINDIE_LOG_TO_LEVEL = debug
```

CANN日志收集|默认输出路径为"~/ascend"

```bash
export ASCEND_GLOBAL_LOG_TO_LEVEL = 1
```

权重路径权限问题
注意保证权重路径是可用的，执行以下命令修改权限，注意是整个父级目录的权限：

```bash
chown -R HwHiAiUser:HwHiAiUser {/path-to-weights}
chmod -R 750 {/path-to-weights}
```