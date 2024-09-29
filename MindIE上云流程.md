[当前文档访问路径](https://ai-fae.readthedocs.io/zh-cn/latest/MindIE上云流程.html)  

- 本次以HF上的qwen1.5 7b chat模型为例，CANN 8.0RC2，MindIE 1.0.RC2，python3.10.12
- 云上服务器使用专属资源池，驱动版本24.1.rc1
# 1 基础镜像
使用杭州AICC基础镜像进行制作 `hzaicc-makeimages-base:v1.0`

# 2 启动容器
构建容器，并且挂载个人目录到容器中，个人目录中有已经下载好的CANN和MindIE安装包，以及qwen1.5 chat模型文件。
```bash
docker run -it -u ma-user \
--privileged=true \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
--ipc=host \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver  \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /opt/data2/wangfeng:/home/ma-user/wangfeng/ \
--name mindie_modelarts \
--entrypoint=/bin/bash \
hzaicc-makeimages-base:v1.0
```

# 3 安装依赖
## 3.1 CANN+MindIE
CANN需要按顺序安装toolkit, kernel, nnal加速库  
安装文件可从[Ascend社区版资源下载](https://www.hiascend.cn/developer/download/community/result?module=ie+pt+cann)页面获取，cpu架构选择AArch64，格式选择run  
需要下载`Ascend-mindie_1.0.RC2_linux-aarch64.run`, `Ascend-cann-toolkit_8.0.RC2_linux-aarch64.run`, `Ascend-cann-kernels-910b_8.0.RC2_linux.run`, `Ascend-cann-nnal_8.0.RC2_linux-aarch64.run`这四个包
- MindIE ATB Models 也需要获取，目前需要在下载页面申请  
- 执行安装脚本，安装目录选择 /home/ma-user/Ascend/ 防止可能出现的权限问题
- 以下操作都由ma-user用户进行操作安装
```bash
# toolkit
./Ascend-cann-toolkit_8.0.RC2_linux-aarch64.run --full --install-path=/home/ma-user/Ascend/ --quiet
echo 'source /home/ma-user/Ascend/ascend-toolkit/set_env.sh' >> ~/.bashrc
source /home/ma-user/Ascend/ascend-toolkit/set_env.sh

# kernel
./Ascend-cann-kernels-910b_8.0.RC2_linux.run --install --install-path=/home/ma-user/Ascend/ --quiet
source /home/ma-user/Ascend/ascend-toolkit/set_env.sh

# nnal
./Ascend-cann-nnal_8.0.RC2_linux-aarch64.run --install --install-path=/home/ma-user/Ascend/ --quiet
echo 'source /home/ma-user/Ascend/nnal/atb/set_env.sh' >> ~/.bashrc
source /home/ma-user/Ascend/nnal/atb/set_env.sh

# mindie
./Ascend-mindie_1.0.RC2_linux-aarch64.run --install --install-path=/home/ma-user/Ascend/ --quiet
echo 'source /home/ma-user/Ascend/mindie/set_env.sh' >> ~/.bashrc
source /home/ma-user/Ascend/mindie/set_env.sh

# atb models
mkdir MindIE-LLM
tar -xzf Ascend-mindie-atb-models_1.0.RC2_linux-aarch64_torch2.1.0-abi0.tar.gz -C MindIE-LLM
mv MindIE-LLM /home/ma-user/Ascend/

# pytorch
pip install torch==2.1.0
pip install torch_npu-2.1.0.post6.dev20240716-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# atb models的其他依赖
pip install apex-0.1.dev20240716+ascend-cp310-cp310-linux_aarch64.whl
pip install /home/ma-user/Ascend/MindIE-LLM/atb_llm-0.0.1-py3-none-any.whl
pip install -r /home/ma-user/Ascend/MindIE-LLM/requirements/requirements.txt
pip install -r /home/ma-user/Ascend/MindIE-LLM/requirements/models/requirements_qwen1.5.txt
echo 'export LD_LIBRARY_PATH=/home/ma-user/anaconda3/lib/:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'source /home/ma-user/Ascend/MindIE-LLM/set_env.sh' >> ~/.bashrc
source /home/ma-user/Ascend/MindIE-LLM/set_env.sh

# 下面这行是为了上云后本地端口访问不被转发
echo 'export no_proxy=127.0.0.1,$no_proxy' >> ~/.bashrc

```


# 4 测试ATB Models

```bash
cd /home/ma-user/Ascend/MindIE-LLM
# -m 参数为模型权重的文件夹路径
bash examples/models/qwen/run_fa.sh -m /home/ma-user/wangfeng/qwen15/7b/
```

# 5 测试mindie servece
## 5.1 启动服务
```bash
cd /home/ma-user/Ascend/mindie/latest/mindie-service/conf
vi config.json
# 修改 https false
# 修改seq长度 4096 （可选）
# 修改模型路径 /home/ma-user/wangfeng/qwen15/7b/
cd /home/ma-user/Ascend/mindie/latest/mindie-service
# 启动服务
./bin/mindieservice_daemon
```


## 5.2 测试接口
新开一个容器链接，输入测试文本
```bash
curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{
  "inputs": "My name is Olivier and I",
  "parameters": {
    "best_of": 1,
    "decoder_input_details": false,
    "details": false,
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
返回结果
```bash
[{"generated_text":" am a French photographer living in the Netherlands. I have been working as a professional photographer for over 10 years, and I specialize in portrait photography, lifestyle photography, and commercial photography.\nI am passionate about capturing the unique personality of each individual I"}
```

# 6 Gradio 内网穿透服务
## 6.1 安装
```
pip install gradio==4.44.0
```

## 6.2 测试gradio服务
```python
import gradio as gr
def greet(name):
    return "Hello " + name + "!"
demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch(share=True)
```
一般这一步会无法生成外网访问链接，并出现如下的报错信息
```bash
[ma-user mindie-service]$python 1.py 
Running on local URL:  http://127.0.0.1:7860

Could not create share link. Missing file: /home/ma-user/anaconda3/lib/python3.10/site-packages/gradio/frpc_linux_aarch64_v0.2. 
```
这一步是因为缺少frpc文件，需要编译后复制文件到报错中的指定位置

## 6.3 编译frpc文件
```bash 
clone https://github.com/huggingface/frp
cd frp
# 根据frp文件夹下的go.mod确定go版本
wget https://golang.google.cn/dl/go1.18.10.linux-arm64.tar.gz
tar -C /usr/local -zxvf go1.18.0.linux-arm64.tar.gz
echo 'export GOROOT=/home/ma-user/go' >> /home/ma-user/.bashrc
echo 'export PATH=$PATH:$GOROOT/bin' >> /home/ma-user/.bashrc
echo 'export GOPATH=$HOME/go' >> /home/ma-user/.bashrc
source ~/.bashrc
# go version #验证
# 启用go模块 设置代理，以便国内下载依赖包加速
go env -w GO111MODULE=on
go env -w GOPROXY=https://goproxy.cn,direct
# 编译frpc
make frpc
# 复制frpc文件到之前报错的位置
cp bin/frpc /home/ma-user/anaconda3/lib/python3.10/site-packages/gradio/frpc_linux_aarch64_v0.2
```

## 6.4 修复后输出内容
```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://a7f6d7882947e7747d.gradio.live
```
如果public URL能够正常访问服务，即安装正常

# 7 Gradio转发MindIE端口服务测试
需要启动mindie服务，并复制如下代码后启动Gradio服务

```python demo.py
import gradio as gr
import requests
import json

def generate_text(input_text):
    url = "http://127.0.0.1:1025/v1/chat/completions"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    
    payload = {
         "model": "qwen",
         "messages": [{
                "role": "system",
                "content": "You are a helpful assistant."
            },{
                "role": "user",
                "content": input_text
            }],
        "max_tokens": 500,
        "presence_penalty": 1.03,
        "frequency_penalty": 1.0,
        "temperature": 0.5,
        "top_p": 0.95,
        "stream": False
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()
        print(result)
        return result['choices'][0]['message']['content']
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
访问生成的外网链接，左侧聊天框输入提示词，如果右边能正常返回服务即部署完成  
保存镜像上传到ModelArts测试服务即可