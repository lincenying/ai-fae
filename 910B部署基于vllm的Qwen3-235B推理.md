[当前文档访问路径](https://ai-fae.readthedocs.io/zh-cn/latest/910B部署基于vllm的Qwen3-235B推理.html)

# 1. 环境准备
## 1.1 服务器要求

910B 8卡 x 2

# 2. 配置裸金属相关环境
## 2.1 安装驱动
使用`ssh`连接裸金属后, 执行以下命令:
```bash
npu-smi info # 如果驱动版本是24.1.rc3, 以下安装更新驱动步骤可省略

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
# v0.8.4rc2
wget http://39.171.244.84:30011/vllm/vllm-ascend-0.8.4rc2.tar.gz
docker load -i vllm-ascend-0.8.4rc2.tar.gz
# 或者
docker pull swr.cn-east-3.myhuaweicloud.com/kubesre/quay.io/ascend/vllm-ascend:v0.8.4rc2-openeuler-linux-arm64
# 或者
docker pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/quay.io/ascend/vllm-ascend:v0.8.4rc2-openeuler-linuxarm64
# v0.8.4rc1
# docker pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/quay.io/ascend/vllm-ascend:v0.8.4rc1-openeuler-linuxarm64
# v0.7.3rc2
# docker pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/quay.io/ascend/vllm-ascend:v0.7.3rc2-linuxarm64

# 通过 obsutil 下载镜像, 需执行步骤 2.3
mkdir -p /data/docker_images
cd /data/docker_images
# v0.8.4rc2
obsutil cp obs://docker/vllm-ascend-0.8.4rc2.tar.gz ./vllm-ascend-0.8.4rc2.tar.gz
docker load -i vllm-ascend-0.8.4rc2.tar.gz
# v0.8.4rc1
# obsutil cp obs://docker/vllm-ascend-v0.8.4rc1-openeuler.tar ./vllm-ascend-v0.8.4rc1-openeuler.tar
# docker load -i vllm-ascend-v0.8.4rc1-openeuler.tar
# v0.7.3rc2
# obsutil cp obs://docker/vllm-ascend-v0.7.3rc2.tar.gz ./vllm-ascend-v0.7.3rc2.tar.gz
# docker load -i vllm-ascend-v0.7.3rc2.tar.gz

```

## 3.2 下载模型

使用`魔搭社区`下载模型, 也可以使用`Huggingface`或`魔乐社区`
注意: 两个节点都需要下载, 路径保持一致

```bash
pip install modelscope

cd /data2
# 根据情况下载所需要模型
modelscope download --model Qwen/Qwen3-235B-A22B --local_dir ./Qwen3-235B-A22B

```
或者使用`obsutil`下载 (需执行步骤 2.3)

```bash
# 根据情况下载所需要模型
obsutil cp obs://bigmodel/Qwen3-235B-A22B/ ./Qwen3-235B-A22B/ -f -r -flat

```

## 3.3 启动容器

两个节点分别使用`docker images`查看下载下来的`image`，
使用下面启动命令(参考)：

```bash
docker run -itd \
--name vllm-server-qwen3-235b \
--privileged \
--network host \
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
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /data2:/data2 \
quay.io/ascend/vllm-ascend:v0.8.4rc2-openeuler \
/bin/bash

# -v /data:/data 和 -v /data2:/data2 根据实际情况挂载
# quay.io/ascend/vllm-ascend:v0.8.4rc2-openeuler 根据实际情况, 修改为对应的镜像名:Tag

```

进入容器:
```bash
# docker ps 查看下容器ID 或者使用 容器name
docker exec -it vllm-server-qwen3-235b /bin/bash
```

## 3.4 设置ray集群
```bash
# 主节点
# 当前节点内网ip
echo 'export local_ip=192.168.0.15' >> ~/.bashrc
# 当前节点内网ip对应网卡名称
echo 'export nic_name=enp67s0f0' >> ~/.bashrc

echo 'export HCCL_IF_IP=$local_ip' >> ~/.bashrc
echo 'export GLOO_SOCKET_IFNAME=$nic_name' >> ~/.bashrc
echo 'export TP_SOCKET_IFNAME=$nic_name' >> ~/.bashrc
echo 'export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1 ' >> ~/.bashrc
echo 'export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7' >> ~/.bashrc

source ~/.bashrc
ray start --head --port=6379 --num-gpus=8

# 副节点
# 当前节点内网ip
echo 'export local_ip=192.168.0.21' >> ~/.bashrc
# 当前节点内网ip对应网卡名称
echo 'export nic_name=enp67s0f0' >> ~/.bashrc

echo 'export HCCL_IF_IP=$local_ip' >> ~/.bashrc
echo 'export GLOO_SOCKET_IFNAME=$nic_name' >> ~/.bashrc
echo 'export TP_SOCKET_IFNAME=$nic_name' >> ~/.bashrc
echo 'export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1 ' >> ~/.bashrc
echo 'export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7' >> ~/.bashrc

source ~/.bashrc
ray start --address='192.168.0.15:6379' --num-gpus=8 --node-ip-address=$local_ip
```

# 4. 拉起服务

进入主节点的docker容器

```bash
# VLLM_TEST_ENABLE_EP=1 开启专家并行
# VLLM_USE_V1=1 使用vLLM的V1引擎
VLLM_TEST_ENABLE_EP=1 \
VLLM_USE_V1=0 \
vllm serve /data2/Qwen3-235B-A22B \
    --served-model-name Qwen3-235B-A22B \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 16 \
    --port 8006 \
    --block-size 128 \
    --distributed_executor_backend "ray" \
    --enforce-eager 

```
## 4.1 参数说明

```bash
--model_tag‌：指定模型路径或名称（支持 Hugging Face 仓库或本地路径），例如 vllm serve "gpt-neo-2.7B"。
‌--config‌：通过 YAML 文件加载复杂配置，避免命令行参数冗余。
‌--host 和 --port‌：设置服务地址与端口，默认值为 127.0.0.1:8000。
# NPU与内存优化参数
‌--tensor-parallel-size‌：定义 Tensor 并行度（NPU 数量），用于多卡分布式推理。
‌--pipeline-parallel-size‌：设定流水线并行度，将模型层分配给不同 NPU。
‌--Npu-memory-utilization‌：控制 NPU 内存利用率（0-1），默认值 0.9，适用于内存紧张的场景。
‌--cpu-offload-gb‌：将部分模型数据卸载到 CPU 内存（单位 GB），缓解 NPU 内存压力。
‌--dtype/--fp16‌：设置计算精度（float16、bfloat16 或启用 FP16），降低内存占用并加速推理。
# 模型性能与资源限制
‌--max-model-len‌：指定最大输入序列长度（token 数），防止内存溢出。
‌--max-num-batched-tokens‌：控制每批次处理的 token 上限，优化吞吐量。
‌--max-concurrent-requests‌：限制并发请求数，避免资源过载。
# 其他实用参数
‌--model-cache-dir‌：指定模型缓存目录，便于重复使用或共享权重文件。
‌--log-level‌：调整日志详细程度（DEBUG/INFO/WARNING 等），便于调试。
‌--swap-space‌：分配 NPU 交换空间（GB），提升多卡场景下的内存效率。
--block-size：连续 token 块的 token 块大小
--distributed_executor_backend：用于指定分布式执行器的后端类型
```
完整参数见:
https://vllm.hyper.ai/docs/inference-and-serving/engine_args

## 4.3 openai接口

另外新起一个窗口，输入命令发送POST请求：
```bash
curl -H "Accept: application/json" -H "Content-Type: application/json" -X POST -d '{
    "model": "Qwen3-235B-A22B",
    "messages": [{
        "role": "user",
        "content": "你是谁"
    }],
    "max_tokens": 512,
    "stream": false
}' http://192.168.0.15:8006/v1/chat/completions

```
