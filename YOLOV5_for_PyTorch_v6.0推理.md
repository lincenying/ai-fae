镜像: py3.8_torch2.1.0_cann7.0.rc1_euler-2.8.3-aarch64:202311040618
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
cd ./modelzoo-GPL/built-in/AscendIE/TorchAIE/built-in/cv/detection/Yolov5
```

```bash
# 获取Pytorch源码
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout v6.0

# 将推理部署代码拷贝到yolov5源码相应目录下
cp -r ../common ./
cp ../aie_compile.py ./
cp ../aie_val.py ./
cp ../model.yaml ./
# 根据版本应用对应的补丁
git apply ./common/patches/v6.0.patch

```

```bash
# 安装依赖
pip install numpy==1.23
pip install tqdm
pip install opencv-python
pip install pandas==2.0.2
pip install requests
pip install pyyaml
pip install Pillow==9.5
pip install matplotlib
pip install seaborn
pip install pycocotools

```

```bash
# 下载obsutil
cd /home/ma-user/work/
wget https://obsbrowser.obs.cn-east-292.mygaoxinai.com/obsutil_hcso_linux_arm64_5.3.4.tar.gz
tar -zxvf obsutil_hcso_linux_arm64_5.3.4.tar.gz
chmod +x ./obsutil_linux_arm64_5.3.4/obsutil
./obsutil_linux_arm64_5.3.4/obsutil config -i=###替换成AK### -k=###替换成SK### -e=obs.cn-east-292.mygaoxinai.com

# 下载coco数据集
./obsutil_linux_arm64_5.3.4/obsutil cp obs://temp-zjw/datasets/coco2017.zip ./
unzip coco2017.zip
mv ./coco2017 ./coco
mv ./coco /home/ma-user/work/modelzoo-GPL/built-in/AscendIE/TorchAIE/built-in/cv/detection/Yolov5/yolov5
cp /home/ma-user/work/modelzoo-GPL/built-in/PyTorch/Official/cv/object_detection/Yolov5_for_PyTorch_v6.0/coco/* /home/ma-user/work/modelzoo-GPL/built-in/AscendIE/TorchAIE/built-in/cv/detection/Yolov5/yolov5/coco

cd /home/ma-user/work/modelzoo-GPL/built-in/AscendIE/TorchAIE/built-in/cv/detection/Yolov5/yolov5/coco
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