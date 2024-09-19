- 本次以HF上的qwen1.5 7b chat模型为例，CANN 8.0RC2，MindIE 1.0.RC2，python3.10.12

# 1 基础镜像
使用杭州AICC基础镜像进行制作 `hzaicc-makeimages-base:v1.0`

# 2 启动容器
构建挂载4卡的容器，并且挂载个人目录到容器中，个人目录中有已经下载好的CANN和MindIE安装包，以及qwen1.5 chat模型文件。
```bash
docker run -it -u root \
--privileged=true \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
--ipc=host \
--net=host \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver  \
-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /home/wangfeng:/home/ma-user/wangfeng/ \
--name mindie-test \
--entrypoint=/bin/bash \
hzaicc-makeimages-base:v1.0
```

# 3 安装依赖
# 3.1 CANN+MindIE
CANN需要按顺序安装toolkit, kernel, nnal加速库  
安装文件可从[Ascend社区版资源下载](https://www.hiascend.cn/developer/download/community/result?module=ie+pt+cann)页面获取，cpu架构选择AArch64，格式选择run  
需要下载`Ascend-mindie_1.0.RC2_linux-aarch64.run`, `Ascend-cann-toolkit_8.0.RC2_linux-aarch64.run`, `Ascend-cann-kernels-910b_8.0.RC2_linux.run`, `Ascend-cann-nnal_8.0.RC2_linux-aarch64.run`这四个包
- MindIE ATB Models 也需要获取，目前需要在下载页面申请，也可以直接联系FAE索取软件包  
执行安装脚本
```bash
chmod +x Ascend-cann-toolkit_8.0.RC2_linux-aarch64.run
chmod +x Ascend-cann-kernels-910b_8.0.RC2_linux.run
chmod +x Ascend-cann-nnal_8.0.RC2_linux-aarch64.run
chmod +x Ascend-mindie_1.0.RC2_linux-aarch64.run
./Ascend-cann-toolkit_8.0.RC2_linux-aarch64.run --install --install-for-all --quiet
source ~/.bashrc
./Ascend-cann-kernels-910b_8.0.RC2_linux.run --install --install-for-all --quiet
source ~/.bashrc
./Ascend-cann-nnal_8.0.RC2_linux-aarch64.run --install --quiet
echo 'source /usr/local/Ascend/nnal/atb/set_env.sh' >> /home/ma-user/.bashrc
source ~/.bashrc
./Ascend-mindie_1.0.RC2_linux-aarch64.run --install --quiet
echo 'source /usr/local/Ascend/mindie/set_env.sh' >> /home/ma-user/.bashrc
# /usr/local/Ascend/driver/lib64/driver/
source ~/.bashrc
```
# 3.2 安装torch
```bash 
# 获取torch npu包
# wget https://gitee.com/ascend/pytorch/releases/download/v6.0.rc2-pytorch2.1.0/torch_npu-2.1.0.post6-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
# 安装torch和torch npu
pip install torch==2.1.0
pip install torch_npu-2.1.0.post6-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

# 3.3 ATB Models
将下载的ATB Models解压，并复制到/usr/local/Ascend/目录下
```bash
mkdir MindIE-LLM
tar -xzf Ascend-mindie-atb-models_1.0.RC2_linux-aarch64_torch2.1.0-abi0.tar.gz -C MindIE-LLM
mv MindIE-LLM /usr/local/Ascend/
pip install /usr/local/Ascend/MindIE-LLM/atb_llm-0.0.1-py3-none-any.whl
pip install -r /usr/local/Ascend/MindIE-LLM/requirements/requirements.txt
pip install -r /usr/local/Ascend/MindIE-LLM/requirements/models/requirements_qwen1.5.txt
echo 'export LD_LIBRARY_PATH=/usr/local/anaconda3/lib/:$LD_LIBRARY_PATH' >> /home/ma-user/.bashrc
echo 'source /usr/local/Ascend/MindIE-LLM/set_env.sh' >> /home/ma-user/.bashrc
source ~/.bashrc
```

# 4 运行ATB
```bash
cd /usr/local/Ascend/mindie/latest/mindie-service/conf
vi config.json
# 修改 https false
# 修改seq长度 4096
# 修改模型路径 /home/ma-user/wangfeng/model_from_hf/qwen1.5/7b/chat/chat
cd /usr/local/Ascend/mindie/latest/mindie-service
./bin/mindieservice_daemon
```

# 5 测试
新开一个容器链接，输入测试文本
```bash
curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{
  "inputs": "My name is Olivier and I",
  "parameters": {
    "best_of": 1,
    "decoder_input_details": true,
    "details": true,
    "do_sample": true,
    "max_new_tokens": 50,
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
  "stream": false}' http://127.0.0.1:1025/
```

# 6 安装gradio
```
pip install gradio
```
pip安装的gradio无法进行外网分享，缺少frpc文件，需要用go编译，[解决办法参考](https://github.com/gradio-app/gradio/issues/6053)
```bash 
clone https://github.com/huggingface/frp
cd frp
# 根据frp文件夹下的go.mod确定go版本
wget https://golang.google.cn/dl/go1.18.10.linux-arm64.tar.gz
tar -C /usr/local -zxvf go1.23.0.linux-arm64.tar.gz
echo 'export GOROOT=/usr/local/go' >> /home/ma-user/.bashrc
echo 'export PATH=$PATH:$GOROOT/bin' >> /home/ma-user/.bashrc
echo 'export GOPATH=$HOME/go' >> /home/ma-user/.bashrc
source ~/.bashrc
# go version #验证
# 启用go模块 设置代理，以便国内下载依赖包加速
go env -w GO111MODULE=on
go env -w GOPROXY=https://goproxy.cn,direct
# 编译frpc
make frpc
cp bin/frpc /usr/local/anaconda3/lib/python3.10/site-packages/gradio/frpc_linux_aarch64_v0.2
```

# 7 测试gradio demo
```python
import gradio as gr

def greet(name):
    return "Hello " + name + "!"


demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch(share=True)
```
输出内容
```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://a7f6d7882947e7747d.gradio.live
```
如果访问public URL，能够正常访问服务，即安装正常；有其他问题需自行排查

# 8 启动服务
需要启动两个容器连接，一个用于mindie服务，一个用于gradio服务
- MindIE
```bash
cd /usr/local/Ascend/mindie/latest/mindie-service
./bin/mindieservice_daemon
```
- Gradio
```python demo.py
import gradio as gr
import requests
import json

def generate_text(input_text):
    url = "http://127.0.0.1:1025/"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": input_text,
        "parameters": {
            "best_of": 1,
            "decoder_input_details": True,
            "details": True,
            "do_sample": True,
            "max_new_tokens": 50,
            "repetition_penalty": 1.03,
            "return_full_text": False,
            "seed": None,
            "stop": ["photographer"],
            "temperature": 0.5,
            "top_k": 10,
            "top_n_tokens": 5,
            "top_p": 0.95,
            "truncate": None,
            "typical_p": 0.95,
            "watermark": True
        },
        "stream": False
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()
        print(result)
        return result[0]["generated_text"]
    else:
        print(response)
        return f"Error: {response.status_code}"
        

demo = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(label="Enter your text"),
    outputs=gr.Textbox(label="Generated Text"),
    title="Text Generation Demo"
)

demo.launch(share=True)
```


