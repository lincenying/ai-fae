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
npu-smi info # 如果驱动版本是24.1.rc3, 以下安装更新驱动步骤可省略

yum update -y
yum install wget -y

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

# 3. 配置分布式通信
## 3.1 检查机器网络情况

```bash
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
mkdir -p /data/hm
cd /data/hm
wget  http://39.171.244.84:30011/DistributedCommunication/hccl_tools.py 
python ./hcc_tools.py

```
在你当前目录下生成一个 hccl_xx_(当前物理机ip).json 的文件
将其他的机器生成的hccl.json文件放在主物理机的统一文件夹中，使用下面命令获取合并脚本。
```bash
wget  http://39.171.244.84:30011/DistributedCommunication/merge_hccl.py
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
-v /data2:/data2 \
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
export MIES_CONTAINER_IP=192.168.0.20 # (物理机中使用ifconfig 查看)
export RANKTABLEFILE=/data/hm/hccl_2s_16p.json # (rank_table_file.json 的路径,生成详见第3节)
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
# 替换IP
sed -i 's/"ipAddress"[[:space:]]*:[[:space:]]*".*",/"ipAddress" : "192.168.0.20",/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 替换端口
sed -i 's/"port"[[:space:]]*:[[:space:]]*.*,/"port" : 1025,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 关闭https
sed -i 's/"httpsEnabled"[[:space:]]*:[[:space:]]*.*,/"httpsEnabled" : false,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 开启开启多机推理
sed -i 's/"multiNodesInferEnabled"[[:space:]]*:[[:space:]]*.*,/"multiNodesInferEnabled" : true,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 若不需要安全认证，则将以下两个参数设为false
sed -i 's/"interCommTLSEnabled"[[:space:]]*:[[:space:]]*.*,/"interCommTLSEnabled" : false,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
sed -i 's/"interNodeTLSEnabled"[[:space:]]*:[[:space:]]*.*,/"interNodeTLSEnabled" : false,/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 启用几卡
sed -i 's/"npuDeviceIds"[[:space:]]*:[[:space:]]*\[\[.*\]\],/"npuDeviceIds" : [[0,1,2,3,4,5,6,7]],/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 替换模型名称
sed -i 's/"modelName"[[:space:]]*:[[:space:]]*".*",/"modelName" : "DeepSeek-V3-0324",/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 替换模型路径
sed -i 's/"modelWeightPath"[[:space:]]*:[[:space:]]*".*",/"modelWeightPath" : "\/data2\/DeepSeek-V3-0324-w8a8",/' /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
# 启用几卡
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
chmod -R 750 /data2/DeepSeek-V3-0324-w8a8/
chmod 640 /data/hm/hccl_2s_16p.json

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