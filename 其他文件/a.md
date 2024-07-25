镜像: pytorch2.1.0_cann8.0.rc1.alpha002_py3.9_euler2.8.3_910b:v8
规格: Ascend: 1*ascend-d910b|CPU: 24核 192GB

1. 上传init_env.sh脚本

2. 执行脚本
```bash
cd ~/work
bash init_env.sh
```

3. 设置算子开发所需环境变量
```bash
source ~/.bashrc
```

4. 上传 SinhCustom20240724.zip 代码包到`~/work`目录

```bash
unzip SinhCustom20240724.zip
```

5. 修改代码
文件: SinhCustom/SinhCustom/CMakePresets.json
```json
{
    "ASCEND_COMPUTE_UNIT": {
        "type": "STRING",
        "value": "ascend910b" // 改成对应型号
    },
    "ASCEND_CANN_PACKAGE_PATH": {
        "type": "PATH",
        "value": "/usr/local/Ascend/ascend-toolkit/latest" //  改成对应路径
    },
}
```
文件: SinhCustom/op_host/sinh_custom.cpp
```cpp
this->AICore().AddConfig("ascend910b"); // 改成对应型号
```

其他文件:
SinhCustom/SinhCustom/op_host/sinh_custom_tiling.h
SinhCustom/SinhCustom/op_host/sinh_custom.cpp
SinhCustom/SinhCustom/op_kernel/sinh_custom.cpp
上面文件, 带注释的地方可适当修改变量名

6. 执行
```bash
cd /home/ma-user/work/SinhCustom/SinhCustom
bash build.sh
```
出现提示：
CPack: - package: /home/ma-user/work/SinhCustom/SinhCustom/build_out/custom_opp_euleros_aarch64.run.json generated.
CPack: - package: /home/ma-user/work/SinhCustom/SinhCustom/build_out/custom_opp_euleros_aarch64.run generated.
即为成功

```bash
cd build_out
./custom_opp_euleros_aarch64.run
```
出现提示:
[runtime] [2024-07-24 09:55:07] copy new ops op_api files ......
[runtime] [2024-07-24 09:55:07] [INFO] no need to upgrade custom.proto files
SUCCESS
即为成功

```bash
cd ../../AclNNInvocation/
bash run.sh
```

出现提示:
#####################################
INFO: you have passed the Precision!
#####################################
即为成功