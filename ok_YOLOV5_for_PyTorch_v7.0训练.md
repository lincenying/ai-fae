镜像: py3.10_torch2.1.0_cann7.0.rc1_euler-2.8.3-aarch64


```bash
# 解决`/usr/libexec/git-core/git-remote-https: relocation error: /lib64/libcurl.so.4: symbol SSLv3_client_method version OPENSSL_1_1_0 not defined in file libssl.so.1.1 with link time reference`

export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
```

```bash
# 通过Git获取代码
cd /home/ma-user/work/
git clone https://gitee.com/ascend/modelzoo-GPL.git
cd /home/ma-user/work/modelzoo-GPL/built-in/PyTorch/Official/cv/object_detection/Yolov5_for_PyTorch_v7.0/
# 安装依赖
pip3 install -r requirements.txt
```

```bash
# 编译安装torchvision
cd /home/ma-user/work/
git clone https://github.com/pytorch/vision.git #根据torch版本选择不同分支
cd vision
# git checkout -b v0.9.1 v0.9.1
python setup.py bdist_wheel
pip3 install dist/*.whl
```

```bash
# 编译安装Opencv-python
cd /home/ma-user/work/
export GIT_SSL_NO_VERIFY=true
git clone https://github.com/opencv/opencv.git
cd opencv
mkdir -p build
cd build

cmake -D BUILD_opencv_python3=yes \
-D BUILD_opencv_python2=no \
-D PYTHON3_EXECUTABLE=/usr/local/python3.7.5/bin/python3.7m \
-D PYTHON3_INCLUDE_DIR=/usr/local/python3.7.5/include/python3.7m \
-D PYTHON3_LIBRARY=/usr/local/python3.7.5/lib/libpython3.7m.so \
-D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/python3.7.5/lib/python3.7/site-packages/numpy/core/include \
-D PYTHON3_PACKAGES_PATH=/usr/local/python3.7.5/lib/python3.7/site-packages \
-D PYTHON3_DEFAULT_EXECUTABLE=/usr/local/python3.7.5/bin/python3.7m ..

make -j$nproc
sudo make install

# 如上面安装失败
pip install opencv-python-headless
```

```bash
# 下载obsutil
cd /home/ma-user/work/
wget https://obs-community.obs.cn-north-1.myhuaweicloud.com/obsutil/current/obsutil_linux_arm64.tar.gz
tar -zxvf obsutil_linux_arm64.tar.gz
chmod +x ./obsutil_linux_arm64_5.5.12/obsutil
ln ./obsutil_linux_arm64_5.5.12/obsutil obsutil
/home/ma-user/work/obsutil config -i=###替换成AK### -k=###替换成SK### -e=obs.cn-east-292.mygaoxinai.com

# 通过obsutil下载coco数据源
cd /home/ma-user/work/
./obsutil_linux_arm64_5.3.4/obsutil cp obs://temp-zjw/datasets/coco2017.zip ./
unzip coco2017.zip
mkdir coco
mv coco2017 /home/ma-user/work/modelzoo-GPL/coco
cp /home/ma-user/work/modelzoo-GPL/built-in/PyTorch/Official/cv/object_detection/Yolov5_for_PyTorch_v7.0/cocofile/* /home/ma-user/work/modelzoo-GPL/coco/
cd /home/ma-user/work/modelzoo-GPL/coco/
python3 coco2yolo.py
```

如果报 `ModuleNotFoundError: No module named 'pycocotools'`
```bash
pip3 install pycocotools
python3 coco2yolo.py
```

```bash
cd /home/ma-user/work/modelzoo-GPL/built-in/PyTorch/Official/cv/object_detection/Yolov5_for_PyTorch_v7.0/
ln -s /home/ma-user/work/modelzoo-GPL/coco coco
```

```bash
# 安装可能缺失的依赖
pip3 uninstall pandas
pip3 install pandas
# python3.8版本安装8.12.0版本IPython
pip3 install IPython==8.12.0
pip3 install seaborn
pip3 install tensorboard
```

**可能需要, 也可能不需要的操作**
```bash
vi ./test/train_yolov5s_full_1p.sh
```
```yaml
Network="yolov5s_v7.0"
# 改成
Network="Yolov5_for_PyTorch_v7.0"
```

```yaml
python3 -u train.py \
    --data ./data/coco.yaml \
    --cfg yolov5s.yaml \
# 改成
python3 -u train.py \
    --data ./data/coco.yaml \
    --cfg ./models/yolov5s.yaml \
```

```yaml
python3 val.py \
    --weights yolov5s.pt \
    --data coco.yaml \
# 改成
python3 val.py \
    --weights yolov5s.pt \
    --data ./data/coco.yaml \
```

运行训练脚本
该模型支持单机单卡训练和单机8卡训练。

单机单卡训练

```bash
cd /home/ma-user/work/modelzoo-GPL/built-in/PyTorch/Official/cv/object_detection/Yolov5_for_PyTorch_v7.0/
bash ./test/train_yolo5s_full_1p.sh   # 1p精度    
bash ./test/train_yolo5s_performance_1p.sh   # 1p性能
```
单机8卡训练
```bash
bash ./test/train_yolo5s_full_8p.sh   # 8p精度    
bash ./test/train_yolo5s_performance_8p.sh   # 8p性能
```
多机多卡训练指令
```bash
bash test/train_yolov7_cluster.sh --nnodes=机器数量 --node_rank=机器序号(0,1,2...) --master_addr=主机服务器地址 --master_port=主机服务器端口号
```
ps:脚本默认为8卡，若使用自定义卡数，继续在上面命令后添加 --device_number=每台机器使用卡数 --head_rank=起始卡号，例如分别为4、0时，代表使用0-3卡训练。

--epochs传入训练周期数，默认300， --batch_size传入模型total batch size，可以以单卡batch_size=64做参考设置。

模型评估。
```bash
bash ./test/train_yolov5s_eval_1p.sh 
```
模型训练脚本参数说明如下。

公共参数：
--device                            //训练指定训练用卡
--img-size                          //图像大小
--data                              //训练所需的yaml文件
--cfg                               //训练过程中涉及的参数配置文件
--weights                           //权重
--batch-size                        //训练批次大小
--epochs                            //重复训练次数，默认：300