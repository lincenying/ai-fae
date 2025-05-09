[当前文档访问路径](https://ai-fae.readthedocs.io/zh-cn/latest/910B部署基于MindIE的Qwen3推理.html)

# 1. 环境准备
## 1.1 服务器要求

910B

# 2. 配置裸金属相关环境
## 2.1 安装驱动
使用`ssh`连接裸金属后, 执行以下命令:
```bash
npu-smi info # 如果驱动版本是25.0.rc1.b080, 以下安装更新驱动步骤可省略

# 如果操作系统是Ubuntu 以下步骤可省略 ↓↓↓↓
yum update -y
yum install wget -y
# 如果操作系统是Ubuntu 以上步骤可省略 ↑↑↑↑

mkdir -p /data/drivers
cd /data/drivers
# 官网下载地址: https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=32&cann=8.0.0.beta1&driver=Ascend+HDK+24.1.RC3
wget http://39.171.244.84:30011/drivers/HDK%2025.0.RC1/Ascend-hdk-910b-npu-driver_25.0.rc1.b080_linux-aarch64.run
wget http://39.171.244.84:30011/drivers/HDK%2025.0.RC1/Ascend-hdk-910b-npu-firmware_7.7.t20.0.b200.run
chmod +x Ascend-hdk-910b-npu-driver_25.0.rc1.b080_linux-aarch64.run Ascend-hdk-910b-npu-firmware_7.7.t20.0.b200.run

# 更新驱动
./Ascend-hdk-910b-npu-driver_25.0.rc1.b080_linux-aarch64.run  --upgrade
# 安装驱动
./Ascend-hdk-910b-npu-firmware_7.7.t20.0.b200.run --full

```
## 2.2 安装docker

### 2.2.1 方法1, 使用yum安装
```bash
yum install docker-ce docker-ce-cli containerd.io

```
### 2.2.2 方法2, 通过二进制包手动安装
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
# 通过 docker pull 下载镜像
docker pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/thxcode/mindie:2.0.T17-800I-A2-py311-openeuler24.03-lts-linuxarm64

# 通过 obsutil 下载镜像, 需要执行步骤 2.3
cd /data2
obsutil cp obs://docker/mindie_2.0.T17.B010-800I-A2-py3.11-openeuler24.03-lts-aarch64.tar.gz ./

docker load -i mindie_2.0.T17.B010-800I-A2-py3.11-openeuler24.03-lts-aarch64.tar.gz

```

## 3.2 下载模型

使用`魔搭社区`下载模型, 也可以使用`Huggingface`或`魔乐社区`

```bash
pip install modelscope

cd /data
# 根据情况下载所需要模型
modelscope download --model Qwen/Qwen3-32B --local_dir ./Qwen/Qwen3-32B
modelscope download --model Qwen/Qwen3-14B --local_dir ./Qwen/Qwen3-14B
modelscope download --model Qwen/Qwen3-8B --local_dir ./Qwen/Qwen3-8B

```
或者使用`obsutil`下载, 需执行步骤 2.3

```bash
cd /data

# 根据情况下载所需要模型
obsutil cp obs://bigmodel/Qwen3/Qwen3-32B/ ./Qwen3-32B/ -f -r -flat
obsutil cp obs://bigmodel/Qwen3/Qwen3-14B/ ./Qwen3-14B/ -f -r -flat
obsutil cp obs://bigmodel/Qwen3/Qwen3-8B/ ./Qwen3-8B/ -f -r -flat

```

## 3.3 启动容器

使用`docker images`查看下载下来的`image`的ID，
使用下面启动命令(参考)：

```bash
docker run -itd --privileged  --name=mindie-server-qwen3-32b --net=host \
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
-v /data2:/data2 \
mindie:2.0.T17.B010-800I-A2-py3.11-openeuler24.03-lts-aarch64 \
/bin/bash

```

进入容器:
```bash
# docker ps 查看下容器ID 或者使用 容器name
docker exec -it mindie-server-qwen3-32b /bin/bash
```

## 3.4 升级依赖
```bash
pip install transformers==4.51.0
```

# 4. MindIE服务化启动

## 4.1 修改服务化参数

```bash
cd /usr/local/Ascend/mindie/latest/mindie-service
vi conf/config.json

```
修改以下带注释的参数:
```json
{
    "ServerConfig" :
    {
        "ipAddress" : "192.168.0.20",
        "managementIpAddress" : "127.0.0.2",
        "port" : 1025,
        "managementPort" : 1026,
        "metricsPort" : 1027,
        "httpsEnabled" : false, // 关闭https
    },

    "BackendConfig" : {
        "npuDeviceIds" : [[0,1,2,3]], // 启用几卡推理，910A推荐4卡
        "ModelDeployConfig" :
        {
            "maxSeqLen" : 25600, // 1
            "maxInputTokenLen" : 20480, // 2
            "truncation" : true,
            "ModelConfig" : [
                {
                    "modelInstanceType" : "Standard",
                    "modelName" : "Qwen3-32B",
                    "modelWeightPath" : "/data2/Qwen3-32B",
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
# 修改模型文件夹权限
chmod -R 750 /data2/Qwen3-32B
# 替换IP (192.168.0.23 替换成 本机对应的IP)
sed -i 's/"ipAddress"[[:space:]]*:[[:space:]]*".*",/"ipAddress" : "192.168.0.20",/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 替换端口 (2025 替换成需要监听的端口)
sed -i 's/"port"[[:space:]]*:[[:space:]]*.*,/"port" : 1025,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 关闭https (false 为关闭, true为开启)
sed -i 's/"httpsEnabled"[[:space:]]*:[[:space:]]*.*,/"httpsEnabled" : false,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 启用几卡 ([[0,1,2,3,4,5,6,7]] 为8卡, [[0,1,2,3]] 则为4卡, 根据需要修改)
sed -i 's/"npuDeviceIds"[[:space:]]*:[[:space:]]*\[\[.*\]\],/"npuDeviceIds" : [[0,1,2,3]],/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 替换模型名称 (这个命名影响服务启动, 但是推理时模型名称需要和这个对应)
sed -i 's/"modelName"[[:space:]]*:[[:space:]]*".*",/"modelName" : "Qwen3-32B",/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 替换模型路径 (模型权重的绝对路径)
sed -i 's/"modelWeightPath"[[:space:]]*:[[:space:]]*".*",/"modelWeightPath" : "\/data2\/Qwen3-32B",/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 启用几卡 (4 代表4卡, 需要和上面的 npuDeviceIds 保持一致)
sed -i 's/"worldSize"[[:space:]]*:[[:space:]]*.*,/"worldSize" : 4,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 其他修改
sed -i 's/"maxPrefillBatchSize"[[:space:]]*:[[:space:]]*.*,/"maxPrefillBatchSize" : 1,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
sed -i 's/"maxSeqLen"[[:space:]]*:[[:space:]]*.*,/"maxSeqLen" : 25600,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
sed -i 's/"maxPrefillTokens"[[:space:]]*:[[:space:]]*.*,/"maxPrefillTokens" : 25600,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
sed -i 's/"maxInputTokenLen"[[:space:]]*:[[:space:]]*.*,/"maxInputTokenLen" : 20480,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
sed -i 's/"maxBatchSize"[[:space:]]*:[[:space:]]*.*,/"maxBatchSize" : 50,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
sed -i 's/"maxIterTimes"[[:space:]]*:[[:space:]]*.*,/"maxIterTimes" : 20480,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 查看结果
cat /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json | grep -E 'ipAddress|port|httpsEnabled|multiNodesInferEnabled|interCommTLSEnabled|interNodeTLSEnabled|npuDeviceIds|modelName|modelWeightPath|worldSize|maxPrefillBatchSize|maxSeqLen|maxPrefillTokens|maxInputTokenLen|maxBatchSize|maxIterTimes'


```

## 4.2 拉起服务
```bash
# 解决权重加载过慢问题
export OMP_NUM_THREADS=1
# 拉起服务化
cd /usr/local/Ascend/mindie/latest/mindie-service/
./bin/mindieservice_daemon

# 如果启动失败, 可以执行下面两行命令开启debug, 然后执行启动mindie
export MINDIE_LOG_LEVEL=debug
export MINDIE_LOG_TO_STDOUT=1

# 开启debug调试启动成功后, 先停了服务, 执行下面两行命令关闭debug, 然后执行启动mindie
unset MINDIE_LOG_LEVEL
unset MINDIE_LOG_TO_STDOUT

```

## 4.3 openai接口

另外新起一个窗口，输入命令发送POST请求：
```bash

curl -H "Accept: application/json" -H "Content-Type: application/json" -X POST -d '{
    "model": "Qwen3-32B",
    "messages": [{
        "role": "user",
        "content": "你是谁"
    }],
    "max_tokens": 512,
    "stream": false
}' http://192.168.0.20:1025/v1/chat/completions

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