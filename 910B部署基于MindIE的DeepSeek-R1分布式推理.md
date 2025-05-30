[当前文档访问路径](https://ai-fae.readthedocs.io/zh-cn/latest/910B部署基于MindIE的DeepSeek-R1分布式推理.html)

# 1. 环境准备
## 1.1 服务器要求

Deepseek-R1 16浮点权重需要至少4台atlas 800 I A2 (8*64GB)服务器 w8A8量化权重至少需要两台atlas 800I A2(8*64GB)服务器, deepseekr1量化需使用mindie:2.0.T3版本。
Deepseek-v3 16浮点权重需要至少4台atlas 800 I A2 (8*64GB)服务器 w8A8量化权重至少需要两台atlas 800I A2(8*64GB)服务器。

|  软件   | 版本  |
|  ----  | ----  | 
| cann-toolkit  | 8.0.T63 |
| cann-kernels  | 8.0.T63 |
| cann-nnal  | 8.0.T63.B020 |
| mindie  | 1.0.T71.B020 |

下载链接:
https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f

详情见容器内/usr/local/Ascend

# 2. 配置裸金属相关环境
## 2.1 安装驱动
使用`ssh`连接裸金属后, 执行以下命令:
```bash
npu-smi info # 如果驱动版本大等于24.1.rc3, 以下安装更新驱动步骤可省略

yum update -y
yum install wget -y

# 官网下载地址: https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=32&cann=8.0.0.beta1&driver=Ascend+HDK+24.1.0
wget http://39.171.244.84:30011/drivers/HDK%2024.1.RC3/Ascend-hdk-910b-npu-driver_24.1.rc3_linux-aarch64.run
wget http://39.171.244.84:30011/drivers/HDK%2024.1.RC3/Ascend-hdk-910b-npu-firmware_7.5.0.1.129.run
chmod +x Ascend-hdk-910b-npu-driver_24.1.rc3_linux-aarch64.run Ascend-hdk-910b-npu-firmware_7.5.0.1.129.run

# 更新驱动
./Ascend-hdk-910b-npu-driver_24.1.rc3_linux-aarch64.run  --upgrade
# 安装驱动
./Ascend-hdk-910b-npu-firmware_7.5.0.1.129.run --full

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


# 3. 配置分布式通信
## 3.1 检查机器网络情况

```bash
# 物理机中执行
# 检查物理链接
for i in {0..7}; do hccn_tool -i $i -lldp -g | grep Ifname; done 
# 检查链接情况
for i in {0..7}; do hccn_tool -i $i -link -g; done
# 检查网络健康情况
for i in {0..7}; do hccn_tool -i $i -net_health -g; done
# 查看侦测ip的配置是否正确
for i in {0..7}; do hccn_tool -i $i -netdetect -g; done
# 查看网关是否配置正确
for i in {0..7}; do hccn_tool -i $i -gateway -g; done
# 检查NPU底层tls校验行为一致性，建议全0
for i in {0..7}; do hccn_tool -i $i -tls -g; done | grep switch
# NPU底层tls校验行为置0操作
for i in {0..7}; do hccn_tool -i $i -tls -s enable 0; done
# 获取每张卡的地址
for i in {0..7}; do hccn_tool -i $i -ip -g; done
```

## 3.2 配置rank_table_file.json

每台物理机都需要运行下面命令
```bash
mkdir -p /data/hccl
cd /data/hccl
wget  http://39.171.244.84:30011/DistributedCommunication/hccl_tools.py
# wget  https://gitee.com/lincenying/ai-fea/raw/main/files/hccl_tools.py
python ./hccl_tools.py

```
在你当前目录下生成一个 hccl_xx_(当前物理机ip).json 的文件
将其他的机器生成的hccl.json文件放在主物理机的统一文件夹中，使用下面命令获取合并脚本。
```bash
wget  http://39.171.244.84:30011/DistributedCommunication/merge_hccl.py
# wget  https://gitee.com/lincenying/ai-fea/raw/main/files/merge_hccl.py
# 改成对应的json文件名
python ./merge_hccl.py hccl_8p_01234567_xx.xx.xx.xx.json hccl_8p_01234567_xx.xx.xx.xx.json

```
运行结束后会生成一个总的hccl*.json. 需要在每个`server_id`下加入一行`"container_ip":`和`"server_id":`, value一样。
并且按照ip从大到小排列，且将rank_id手动排序，从0 开始。

然后将整合好的json文件发送到每一个从物理机中。
参考如下格式，配置rank_table_file.json
```jsonc
{
   "server_count": "...", # 总节点数
   # server_list中第一个server为主节点
   "server_list": [
      {
         "device": [
            {
               "device_id": "...", # 当前卡的本机编号，取值范围[0, 本机卡数)
               "device_ip": "...", # 当前卡的ip地址，可通过hccn_tool命令获取
               "rank_id": "..." # 当前卡的全局编号，取值范围[0, 总卡数)
            },
            ...
         ],
         "server_id": "...", # 当前节点的ip地址
         "container_ip": "..." # 容器绑定的物理机ip地址（服务化部署时需要），若无特殊配置，则与server_id相同
      },
      ...
   ],
   "status": "completed",
   "version": "1.0"
}
```

# 4. 准备容器
## 4.1 准备容器

Deepseek-R1 16浮点的镜像包
```bash
# 下载镜像
wget http://39.171.244.84:30011/docker_images/mindie%3A1.0.T71-800I-A2-py311-ubuntu22.04-arm64.tar

# 加载镜像
docker load -i mindie:1.0.T71-800I-A2-py311-ubuntu22.04-arm64.tar
```

如果需要开启量化W8A8的服务，需要使用mindie-2.0.T3的镜像
```bash
# 下载镜像
wget http://39.171.244.84:30011/DistributedCommunication/20T3-800I-A2-py311-openeuler2403-lts.tar

# 加载镜像
docker load -i 20T3-800I-A2-py311-openeuler2403-lts.tar
```

## 4.2 启动容器

使用`docker images`查看下载下来的`image`的ID，
使用下面启动命令(参考)：

```bash
# 如果报 owner not right /usr/bin/runc 1000 错误, 执行:
chown root:root /usr/bin/runc

docker run -itd --privileged  --name=mindie-dsv3-w8a8 --net=host \
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
mindie:2.0.T9.B020-800I-A2-py3.11-openeuler24.03-lts-aarch64

```

或者使用:

```bash
docker run -itd --privileged=true --name=mindie-dsv3-w8a8 --net=host --ipc=host \
--shm-size 500g \
-e ASCEND_VISIBLE_DEVICES=0-7 \
-v /权重路径:/权重路径 \
images_id
```

进入容器:
```bash
# docker ps 查看下容器ID 或者使用 容器name
docker exec -it mindie-dsv3-w8a8 /bin/bash
```

## 4.3 设置变量

```bash
vi ~/.bashrc # 加入下面变量
```

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/atb-models/set_env.sh
source /usr/local/Ascend/mindie/set_env.sh
 # 通信变量
export ATB_LLM_HCCL_ENABLE=1
export ATB_LLM_COMM_BACKEND="hccl"
export HCCL_CONNECT_TIMEOUT=7200
export WORLD_SIZE=16 # 总卡数
export HCCL_EXEC_TIMEOUT=0
export MIES_CONTAINER_IP=192.168.0.20 # (物理机中使用 ifconfig 查看)
export RANKTABLEFILE=/data/hccl/hccl_2s_16p.json # (rank_table_file.json 的路径,生成详见 3.2~3.3)
export HCCL_DETERMINISTIC=true
export NPU_MEMORY_FRACTION=0.95 # 显存比
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True # 虚拟内存
export ASDOPS_LOG_LEVEL=ERROR	# 日志等级
export ASDOPS_LOG_TO_STDOUT=1	# 是否打屏
export MINDIE_LOG_LEVEL=ERROR
export MINDIE_LOG_TO_STDOUT=1	# 是否打屏
export ATB__LOG_LEVEL=ERROR # atb日志等级
export ATB_LOG_TO_STDOUT=1

```

```bash
source ~/.bashrc
```

# 5. 权重下载与转换

# 5.1 权重下载

Deepseek-R1:
https://modelers.cn/models/xieyuxiang/deepseek-r1/tree/main

deepseek-r1 BF-16:
https://modelers.cn/models/xieyuxiang/deepseek-r1-fp16/tree/main

deepseek-v3 BF-16:
https://modelers.cn/models/State_Cloud/DeepSeek-V3-BF16/tree/main

deepseek-r1 W8A8:
https://modelers.cn/models/State_Cloud/DeepSeek-R1-W8A8/tree/main



# 6. MindIE服务化启动
## 6.1 配置服务化环境变量

服务化需要`rank_table_file.json`中配置`container_ip`字段
所有机器的配置应该保持一致，除了环境变量的`MIES_CONTAINER_IP`为本机ip地址。

## 6.2 修改服务化参数

```bash
cd /usr/local/Ascend/mindie/latest/mindie-service
vi conf/config.json

```
修改以下参数:
```jsonc
"httpsEnabled" : false,
"multiNodesInferEnabled" : true, # 开启多机推理
# 若不需要安全认证，则将以下两个参数设为false
"interCommTLSEnabled" : false,
"interNodeTLSEnabled" : false,
"modelName" : "DeepSeek-R1" # 不影响服务化拉起
"modelWeightPath" : "权重路径"

```

或者通过命令行快速修改

```bash
# 替换IP (192.168.0.23 替换成 本机对应的IP)
sed -i 's/"ipAddress"[[:space:]]*:[[:space:]]*".*",/"ipAddress" : "192.168.0.20",/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 替换端口 (2025 替换成需要监听的端口)
sed -i 's/"port"[[:space:]]*:[[:space:]]*.*,/"port" : 1025,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 关闭https (false 为关闭, true为开启)
sed -i 's/"httpsEnabled"[[:space:]]*:[[:space:]]*.*,/"httpsEnabled" : false,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 开启开启多机推理
sed -i 's/"multiNodesInferEnabled"[[:space:]]*:[[:space:]]*.*,/"multiNodesInferEnabled" : true,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 若不需要安全认证，则将以下两个参数设为false
sed -i 's/"interCommTLSEnabled"[[:space:]]*:[[:space:]]*.*,/"interCommTLSEnabled" : false,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
sed -i 's/"interNodeTLSEnabled"[[:space:]]*:[[:space:]]*.*,/"interNodeTLSEnabled" : false,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 启用几卡 ([[0,1,2,3,4,5,6,7]] 为8卡, [[0,1,2,3]] 则为4卡, 根据需要修改)
sed -i 's/"npuDeviceIds"[[:space:]]*:[[:space:]]*\[\[.*\]\],/"npuDeviceIds" : [[0,1,2,3,4,5,6,7]],/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 替换模型名称 (这个命名影响服务启动, 但是推理时模型名称需要和这个对应)
sed -i 's/"modelName"[[:space:]]*:[[:space:]]*".*",/"modelName" : "DeepSeek-V3-0324",/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 替换模型路径 (模型权重的绝对路径)
sed -i 's/"modelWeightPath"[[:space:]]*:[[:space:]]*".*",/"modelWeightPath" : "\/data\/DeepSeek-V3-0324-w8a8",/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 启用几卡 (8 代表8卡, 需要和上面的 npuDeviceIds 保持一致)
sed -i 's/"worldSize"[[:space:]]*:[[:space:]]*.*,/"worldSize" : 8,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 其他修改
sed -i 's/"maxPrefillBatchSize"[[:space:]]*:[[:space:]]*.*,/"maxPrefillBatchSize" : 10,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
sed -i 's/"maxSeqLen"[[:space:]]*:[[:space:]]*.*,/"maxSeqLen" : 24576,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
sed -i 's/"maxInputTokenLen"[[:space:]]*:[[:space:]]*.*,/"maxInputTokenLen" : 16384,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
sed -i 's/"maxPrefillTokens"[[:space:]]*:[[:space:]]*.*,/"maxPrefillTokens" : 16384,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
sed -i 's/"maxBatchSize"[[:space:]]*:[[:space:]]*.*,/"maxBatchSize" : 200,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
sed -i 's/"maxIterTimes"[[:space:]]*:[[:space:]]*.*,/"maxIterTimes" : 8192,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 查看结果
cat /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json | grep -E 'ipAddress|port|httpsEnabled|multiNodesInferEnabled|interCommTLSEnabled|interNodeTLSEnabled|npuDeviceIds|modelName|modelWeightPath|worldSize|maxPrefillBatchSize|maxSeqLen|maxPrefillTokens|maxInputTokenLen|maxBatchSize|maxIterTimes'


```

## 6.3 拉起服务
```bash
chmod -R 750 /data/DeepSeek-V3-0324-w8a8/
chmod 640 /data/hccl/hccl_2s_16p.json

cd /usr/local/Ascend/mindie/latest/mindie-service
./bin/mindieservice_daemon

# 如果启动失败, 可以执行下面两行命令开启debug, 然后执行启动mindie
export MINDIE_LOG_LEVEL=debug
export MINDIE_LOG_TO_STDOUT=1

# 开启debug调试启动成功后, 先停了服务, 执行下面两行命令关闭debug, 然后执行启动mindie
unset MINDIE_LOG_LEVEL
unset MINDIE_LOG_TO_STDOUT

```

## 6.4 openai接口

另外新起一个窗口（也要进入docker），输入命令发送POST请求：
```bash
curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{
  "inputs": "你是谁？",
  "parameters": {
    "best_of": 1,
    "decoder_input_details": false,
    "details": false,
    "do_sample": true,
    "max_new_tokens": 1024,
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
  "stream": false}' http://192.168.0.20:1025/

curl -H "Accept: application/json" -H "Content-Type: application/json" -X POST -d '{
    "model": "DeepSeek-V3-0324",
    "messages": [{
        "role": "user",
        "content": "介绍下杭州西湖"
    }],
    "max_tokens": 512,
    "stream": false
}' http://192.168.0.20:1025/v1/chat/completions
```


# 7. 服务化常见问题
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