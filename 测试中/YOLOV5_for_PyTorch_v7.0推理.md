镜像: py38_pytorch:v4
规格: Ascend: 1*Ascend910|ARM: 24核 96GB

```bash
# 解决`/usr/libexec/git-core/git-remote-https: relocation error: /lib64/libcurl.so.4: symbol SSLv3_client_method version OPENSSL_1_1_0 not defined in file libssl.so.1.1 with link time reference`
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
```

```bash
# 通过Git获取代码, 之前训练过模型, 该步骤可省
cd /home/ma-user/work/
git clone https://gitee.com/ascend/modelzoo-GPL.git
```

```bash
# 获取Pytorch源码
cd /home/ma-user/work/modelzoo-GPL/built-in/ACL_Pytorch/Yolov5_for_Pytorch/
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout v7.0
```

```bash
# 获取OM推理代码
cd /home/ma-user/work/modelzoo-GPL/built-in/ACL_Pytorch/Yolov5_for_Pytorch/
git apply 7.0.patch
```

```bash
# 移动文件
cd /home/ma-user/work/modelzoo-GPL/built-in/ACL_Pytorch/Yolov5_for_Pytorch/
mv common/ ./yolov5
mv model.yaml  ./yolov5/
mv pth2onnx.sh  ./yolov5/
mv onnx2om.sh  ./yolov5/
mv aipp.cfg  ./yolov5/
mv om_val.py  ./yolov5/
mv yolov5_preprocess_aipp.py  ./yolov5/
mv yolov5_preprocess.py  ./yolov5/
mv yolov5_postprocess.py  ./yolov5/
mv requirements.txt  ./yolov5/
```

```bash
# 安装依赖
cd /home/ma-user/work/modelzoo-GPL/built-in/ACL_Pytorch/Yolov5_for_Pytorch/yolov5/
git clone https://gitee.com/ascend/msadvisor.git
cd msadvisor/auto-optimizer
python3 -m pip install --upgrade pip
python3 -m pip install wheel
python3 -m pip install .
cd ../..
pip3 install -r requirements.txt
```

```bash
# 下载obsutil
cd /home/ma-user/work/
wget https://obsbrowser.obs.cn-east-292.mygaoxinai.com/obsutil_hcso_linux_arm64_5.3.4.tar.gz
tar -zxvf obsutil_hcso_linux_arm64_5.3.4.tar.gz
chmod +x ./obsutil_linux_arm64_5.3.4/obsutil
./obsutil config -i=###替换成AK### -k=###替换成SK### -e=obs.cn-east-292.mygaoxinai.com

# 下载coco数据集
./obsutil_linux_arm64_5.3.4/obsutil cp obs://temp-zjw/datasets/coco2017.zip ./
unzip coco2017.zip
mv ./coco2017 ./coco
mv ./coco /home/ma-user/work/modelzoo-GPL/built-in/ACL_Pytorch/Yolov5_for_Pytorch/yolov5/
cp /home/ma-user/work/modelzoo-GPL/built-in/PyTorch/Official/cv/object_detection/Yolov5_for_PyTorch_v7.0/cocofile/* /home/ma-user/work/modelzoo-GPL/built-in/ACL_Pytorch/Yolov5_for_Pytorch/yolov5/coco

cd /home/ma-user/work/modelzoo-GPL/built-in/ACL_Pytorch/Yolov5_for_Pytorch/yolov5/coco
python3 coco2yolo.py
```

```bash
# 下载权重
cd /home/ma-user/work/modelzoo-GPL/built-in/ACL_Pytorch/Yolov5_for_Pytorch/yolov5/
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt

# 安装依赖
pip install IPython==8.12.0

# 将pth转onnx
bash pth2onnx.sh --tag 7.0 --model yolov5l --nms_mode nms_script

# 将onnx转om
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash onnx2om.sh --tag 7.0 --model yolov5l --nms_mode nms_script --bs 4 --soc Ascend910PremiumA

python3 yolov5_preprocess.py --data_path="./coco" --nms_mode nms_script
```