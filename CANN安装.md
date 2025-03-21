[当前文档访问路径](https://ai-fae.readthedocs.io/zh-cn/latest/CANN安装.html)

```bash
docker run -it --name=pytorch2.1.0_cann8_0_rc3_py310 --net=host --ipc=host \
--device=/dev/davinci0 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
-v /usr/local/sbin/:/usr/local/sbin/ \
-v /var/log/npu/slog/:/var/log/npu/slog \
-v /var/log/npu/profiling/:/var/log/npu/profiling \
-v /var/log/npu/dump/:/var/log/npu/dump \
-v /var/log/npu/:/usr/slog \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /opt/data:/home/data \
-v /home/huangming:/home/huangming \
hzaicc-makeimages-base:v2
```

# 1.准备软件包

1. 打开 https://www.hiascend.com/developer/download/community?module=pt+cann&product=4&model=26 根据服务器类型, 选择产品系列

2. 打开 https://www.hiascend.com/developer/download/community/result?module=pt+cann&product=4&model=26, 根据需要的版本查询对应的资源包

如: 
CANN 选 7.0.0.beta1
PyTorch 选 5.0.0.beta1
CPU架构 选 AArch64

3. 根据需要的`pytorch`版本和`python`版本, 下载对应的`torch_npu`包
跳转到gitee后, 可复制下载链接, 使用`wget`下载, 无网络环境则先下载到本地

如:
```bash
wget -O torch_npu-2.1.0.post8-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl \
https://gitee.com/ascend/pytorch/releases/download/v6.0.rc3-pytorch2.1.0/torch_npu-2.1.0.post8-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# wget -O torch_npu-2.1.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl \
# https://gitee.com/ascend/pytorch/releases/download/v5.0.0-pytorch2.1.0/torch_npu-2.1.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

4. 下载 Ascend-cann-toolkit
可以筛选`run`包, 根据服务器是310,910,910b等选择对应的软件包
```bash
wget -O ascend-cann-toolkit-910b_8.0.RC3_linux.run \
https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.RC3/Ascend-cann-toolkit_8.0.RC3_linux-aarch64.run?response-content-type=application/octet-stream

# wget -O ascend-cann-toolkit_7.0.0_linux-aarch64.run \
# https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.0/Ascend-cann-toolkit_7.0.0_linux-aarch64.run?response-content-type=application/octet-stream
```

5. 下载 Ascend-cann-kernels
可以筛选`run`包, 根据服务器是310,910,910b等选择对应的软件包
```bash
wget -O ascend-cann-kernels-910b_8.0.RC3_linux-aarch64.run \
https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.RC3/Ascend-cann-kernels-910b_8.0.RC3_linux-aarch64.run?response-content-type=application/octet-stream

# wget -O ascend-cann-kernels-910b_7.0.0_linux.run \
# https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.0/Ascend-cann-kernels-910b_7.0.0_linux.run?response-content-type=application/octet-stream
```

6. 下载 Ascend-cann-nnal
可以筛选`run`包, 根据服务器是310,910,910b等选择对应的软件包
```bash
wget -O ascend-cann-nnal_8.0.RC3_linux-aarch64.run \
https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.RC3/Ascend-cann-nnal_8.0.RC3_linux-aarch64.run?response-content-type=application/octet-stream

```

# 2.准备安装及运行用户

```bash
# 创建非root用户。
groupadd usergroup     
useradd -g usergroup -d /home/ma-user -m ma-user -s /bin/bash
# 设置非root用户密码。
passwd username
```

# 3.安装开发套件包
```bash
chmod +x ascend-cann-toolkit-910b_7.0.0_linux.run
./ascend-cann-toolkit-910b_7.0.0_linux.run --install

# 默认安装路径如下
# root用户：“/usr/local/Ascend”
# 非root用户：“${HOME}/Ascend”

# 将容器保存成镜像
# 如果对层的大小没有限制, 可以不执行下面docker操作, 直接接着安装二进制算子包
docker commit pytorch2.1.0_cann8_0_rc3_py310 pytorch2.1.0_cann8_0_rc3_py310:20250107_1

# 删除原容器
docker rm pytorch2.1.0_cann8_0_rc3_py310

# 用新保存镜像重新进容器
docker run -it --name=pytorch2.1.0_cann8_0_rc3_py310 --net=host --ipc=host \
--device=/dev/davinci0 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
-v /usr/local/sbin/:/usr/local/sbin/ \
-v /var/log/npu/slog/:/var/log/npu/slog \
-v /var/log/npu/profiling/:/var/log/npu/profiling \
-v /var/log/npu/dump/:/var/log/npu/dump \
-v /var/log/npu/:/usr/slog \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /opt/data:/home/data \
-v /home/huangming:/home/huangming \
pytorch2.1.0_cann8_0_rc3_py310:20250107_1
```

# 4.安装二进制算子包
```bash
chmod +x ascend-cann-kernels-910b_7.0.0_linux.run
./ascend-cann-kernels-910b_7.0.0_linux.run --install

# 默认安装路径如下
# root用户：“/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel”
# 非root用户：“${HOME}/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel”
```

# 5.安装torch_npu
```bash
pip install torch_npu-2.1.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

# 6.验证安装
```bash
cat > test.py <<EOF

import torch
import torch_npu

x = torch.randn(2, 2).npu()
y = torch.randn(2, 2).npu()
z = x.mm(y)

print(z)

EOF

python test.py
```
如果输出正常结果, 则安装成功

# 7.错误解决

## 7.1 cannot allocate memory in static TLS block
```bash
export LD_PRELOAD=/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch.libs/libgomp-4dbbc2f2.so.1.0.0:$LD_PRELOAD
# 路径根据实际报的文件修改
```

## A module that was compiled using NumPy 1.x cannot be run in NumPy 2.1.2 as it may crash.
```bash
# 降级numpy
pip install numpy==1.26.4
```

## ModuleNotFoundError: No module named 'scipy'
```bash
# 安装scipy
pip install scipy
```

```bash
# 保存镜像
docker commit pytorch2.1.0_cann8_0_rc3_py310 pytorch2.1.0_cann8_0_rc3_py310:20250107_2

# 删除原容器
docker rm pytorch2.1.0_cann8_0_rc3_py310
docker rmi pytorch2.1.0_cann8_0_rc3_py310:20250107_1

# 保存镜像
mkdir -p /opt/data/docker_images
docker save pytorch2.1.0_cann8_0_rc3_py310:20250107_2 > /opt/data/docker_images/pytorch2.1.0_cann8_0_rc3_py310.tar

export PATH=/opt/data/huangming/obs_bin:$PATH
obsutil config -i=xxxxxxx -k=xxxxxxx -e=obs.cn-east-292.mygaoxinai.com
obsutil cp /opt/data/docker_images/pytorch2.1.0_cann8_0_rc3_py310.tar obs://zju/image4/
```