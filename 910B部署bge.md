[当前文档访问路径](https://ai-fae.readthedocs.io/zh-cn/latest/910B部署bge.html)


```bash
# 创建存放模型文件夹
mkdir -p /data/model

# 设置敏感路径白名单
export HUB_WHITE_LIST_PATHS=/data/model

# 设置缓存路径
export XDG_CACHE_HOME=/data/model/.cache

# 拉取镜像
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/mis-tei:7.1.RC1-800I-A2-aarch64
```

下载模型:

参考:
http://modelers.cn/docs/zh/openmind-hub-client/0.9/api_reference/download_api.html#snapshot-download

```py
from openmind_hub import snapshot_download
snapshot_download(
    repo_id="MindSDK/bge-m3",
    repo_type="model",
    local_dir="/data/model/bge-m3"
)
```
```py
from openmind_hub import snapshot_download
snapshot_download(
    repo_id="MindSDK/bge-reranker-v2-m3",
    repo_type="model",
    local_dir="/data/model/bge-reranker-v2-m3"
)
```

```bash

cd /data/model

model_dir=/data/model
image_id=f85b838e7160
embedding_model_id=bge-m3
reranker_model_id=bge-reranker-v2-m3
listen_ip=0.0.0.0
embedding_listen_port=55345
reranker_listen_port=55321

# 写入embedding与reranker启动命令 

rm -rf ./mis-tei/temp
mkdir -p ./mis-tei/temp
mkdir -p ./mis-tei/logs
echo "#!/bin/bash" > ./mis-tei/temp/run.sh
echo "" >> ./mis-tei/temp/run.sh
echo "bash start.sh /home/HwHiAiUser/model/$embedding_model_id $listen_ip $embedding_listen_port > logs/embedding.log 2>&1 &" >> ./mis-tei/temp/run.sh
echo "sleep 20" >> ./mis-tei/temp/run.sh
echo "bash start.sh /home/HwHiAiUser/model/$reranker_model_id $listen_ip $reranker_listen_port > logs/reranker.log 2>&1 &" >> ./mis-tei/temp/run.sh
echo "wait" >> ./mis-tei/temp/run.sh
chmod +x ./mis-tei/temp/run.sh

# 启动容器 

docker run -itd --name=tei --net=host --privileged --user root \
-e HOME=/home/HwHiAiUser \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device=/dev/devmm_svm \
--device=/dev/davinci0 \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
-v /usr/local/sbin:/usr/local/sbin:ro \
-v $model_dir:/home/HwHiAiUser/model \
-v $model_dir/mis-tei/temp/run.sh:/home/HwHiAiUser/run.sh \
-v $model_dir/mis-tei/logs:/home/HwHiAiUser/logs \
--entrypoint bash \
$image_id

# 进入容器
docker exec -it tei bash

# 启动服务
bash run.sh
```

```bash
curl http://127.0.0.1:55345/embed \
    -X POST \
    -d '{"inputs":["What is Deep Learning?", "Deep Learning is not..."]}' \
    -H 'Content-Type: application/json'

curl http://127.0.0.1:55321/rerank \
    -X POST \
    -d '{"query":"What is Deep Learning?", "texts": ["Deep Learning is not...", "Deep learning is..."]}' \
    -H 'Content-Type: application/json'

```

FAQ:
```
Error: failed to build prometheus recorder

Caused by:
    failed to create HTTP listener: Address already in use (os error 98)
```

如上错误提示，prometheus 端口被占用, 修改`start.sh`文件, 修改如下代码, 重新指定 `prometheus-port`:
```bash
text-embeddings-router \
      --model-id "${MODEL_DIR}/${MODEL_ID##*/}" \
      --port "${LISTEN_PORT}" \
      --hostname "${LISTEN_IP}" \
      --prometheus-port "9800"
```