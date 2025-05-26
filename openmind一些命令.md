镜像:

```bash
# pytorch
docker pull registry.modelers.cn/base_image/openmind:openeuler-python3.10-cann8.0.rc3.beta1-pytorch2.1.0-openmind1.0.0

# mindspore
docker pull registry.modelers.cn/base_image/openmind:openeuler-python3.10-cann8.0.rc3.beta1-mindspore2.4.0-openmind1.0.0
```

下载模型:

```bash
# 设置敏感路径白名单
export HUB_WHITE_LIST_PATHS=/data01

# 设置缓存路径
export XDG_CACHE_HOME=/data01/.cache
```

下载代码:
```py
from openmind_hub import snapshot_download
snapshot_download(
    repo_id="MindSpore-Lab/Qwen3-1.7B-Base",
    token="0a911143905ec874fa24ae49984ad48915fd7768",
    repo_type="model",
    local_dir="/data01/Qwen3-1.7B-Base",
    local_dir_use_symlinks=False
)
```

创建容器:

```bash
docker run -itd \
--name openmind \
--device /dev/davinci0 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /data01:/data01 \
-ti registry.modelers.cn/base_image/openmind:openeuler-python3.10-cann8.0.rc3.beta1-mindspore2.4.0-openmind1.0.0 /bin/bash
```