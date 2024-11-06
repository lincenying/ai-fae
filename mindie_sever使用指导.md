[当前文档访问路径](https://ai-fae.readthedocs.io/zh-cn/latest/mindie_sever使用指导.html)

docker pull swr.cn-east-292.mygaoxinai.com/huqs/mindie_server_chatgpt_web_910b:20240923_T65

# 1. 环境搭建
## 1.1 启动容器

执行以下命令启动容器，新增映射物理机上存放代码的目录，如物理机上存放代码路径为/home，则将下述的{code_path}替换为/home

```bash
docker run -it --privileged --name=mindie_server_hm --net=host --ipc=host \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
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
mindie_server_chatgpt_web_910b:20240923_T65 \
/bin/bash

```

## 1.2 进入容器
```bash
docker exec -it mindie_server_hm /bin/bash
```

## 1.3 配置环境变量（已写入bashrc，exec进入时自动source）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/mindie/set_env.sh
source /usr/local/Ascend/mindie/latest/mindie-service/set_env.sh
source /opt/atb-models/set_env.sh

```

# 2. 纯模型性能测试
## 2.1 进入模型代码路径
```bash
cd /opt/atb-models/tests/modeltest
```
命令参考：
```
bash run.sh pa_fp16 [performance|full_CEval|full_MMLU|full_BoolQ|full_HumanEval] ([case_pair]) [batch_size] [model_name] ([use_refactor]) [weight_dir] [chip_num] ([max_position_embedding/max_sequence_length])
```
命令实例：
```bash
bash run.sh pa_fp16 performance [[256,256],[512,512],[1024,1024],[2048,2048]] 16 llama True /nfs_share/weights/llama2_13b 8
```
说明:
1. case_pair只在performance场景下接受输入，接收一组或多组输入，格式为[[seq_in_1,seq_out_1],...,[seq_in_n,seq_out_n]], 如[[256,256],[512,512],[1024,1024],[2048,2048]]
2. model_name:当model_name为llama时，须指定use_refactor为True或者False（统一使用True）
4. weight_dir: 权重路径
5. chip_num: 使用的卡数
6. max_position_embedding: 可选参数，不传入则使用config中的默认配置
7. 运行完成后，会在控制台末尾呈现保存数据的文件夹

# 3. Mindie_server服务化能力
## 3.1 修改配置
```bash
cd /usr/local/Ascend/mindie/latest/mindie-service
vim conf/config.json

```
修改服务端口, 权重路径以及worldsize（推理使用的卡数）
```json
{
  "ServerConfig" : {
    "ipAddress" : "127.0.0.1",
    "managementIpAddress": "127.0.0.2",
    "port" : 2025,
    "managementPort" : 2026,
  },
  "BackendConfig": {
      "engineName" : "mindieservice_llm_engine",
      "modelInstanceNumber" : 1,
      "tokenizerProcessNumber" : 8,
      "maxSeqLen" : 4096,
      "npuDeviceIds" : [[0,1,2,3]], // 推理使用的卡数
      "multiNodesInferEnabled" : false,
      "ModelParam" : [
          {
              "modelInstanceType" : "Standard",
              "modelName" : "qwen2_14b",
              "modelWeightPath" : "/home/data/weights/chatglm3-6b/", // 权重路径
              "worldSize" : 4, // 推理使用的卡数
              "cpuMemSize" : 5,
              "npuMemSize" : 8,
              "backendType" : "atb",
              "pluginParams" : ""
          }
      ]
  }
}
```
```json
{
    "ServeParam" :
        {
            "httpsEnabled" : false // 设置为false
        }
}
```

## 3.2 config.json常用参数

| 配置项 | 配置说明 |
|  ----  | ----  |
| maxSeqLen | 最大序列长度。输入的长度+输出的长度<=maxSeqLen，用户根据自己的推理场景选择maxSeqLen。|
| maxBatchSize | 最大decode batch size。|
| npuMemSize | NPU中可以用来申请kv cache的size上限。单位：GB。建议值：8。<br>npuMemSize=（总空闲-权重/tp数）*系数，其中系数取0.8。<br>以llama-65b为例：总显存64GB，空闲状态卡上有3~4GB的占用，<br>llama-65b的总权重为122GB，用8张卡跑，则npuMemSize取值的上限为：(64-4-(122/8))*0.8。|
| worldSize | 启用几张卡推理。|
| npuDeviceIds | 启用哪几张卡。|
| modelWeightPath | 模型权重路径。路径需存在。|
| maxPrefillBatchSize | 最大prefill batch size。该参数主要是在明确需要限制prefill阶段batch size的场景下使用，否则可以设置为与maxBatchSize值相同。|
| maxPrefillTokens | 最大prefill token数量。|
| supportSelectBatch | batch选择策略。<br>false：表示每一轮调度时，优先调度和执行prefill阶段的请求。<br>true：表示每一轮调度时，根据当前prefill与decode请求的数量，自适应调整prefill和decode阶段请求调度和执行的先后顺序。|
| maxIterTimes | 可以进行的decode次数，即一句话最大可生成长度。|
| maxPreemptCount | 每一批次最大可抢占请求的上限，即限制一轮调度最多抢占请求的数量，最大上限为maxBatchSize|

## 3.3 启动daemon服务
```bash
cd /usr/local/Ascend/mindie/latest/mindie-service
./bin/mindieservice_daemon
```
另外新起一个窗口（也要进入docker），输入命令发送POST请求：
```bash
curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{
  "inputs": "变压器油中溶解气体以CH4、C2H4为主要组分时，其故障类型是什么。",
  "parameters": {
    "best_of": 1,
    "decoder_input_details": false,
    "details": false,
    "do_sample": true,
    "max_new_tokens": 1024,
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
  "stream": false}' http://127.0.0.1:2025/
```
看到“generated_text”的返回结果不是乱码，而是正常的一句话。说明运行成功。

# 4. 部署chatgpt-web

## 4.1 安装nodejs

```bash
mkdir -p /usr/local/lib/nodejs
mkdir -p /home/work

cd /home/work/
wget https://mirrors.aliyun.com/nodejs-release/v20.10.0/node-v20.10.0-linux-arm64.tar.gz

tar -zxvf node-v20.10.0-linux-arm64.tar.gz -C /usr/local/lib/nodejs

# 创建 node 软链
ln -s /usr/local/lib/nodejs/node-v20.10.0-linux-arm64/bin/node /usr/bin/node 

# 创建 npm 软链
ln -s /usr/local/lib/nodejs/node-v20.10.0-linux-arm64/bin/npm /usr/bin/npm 

# 创建 npx 软链
ln -s /usr/local/lib/nodejs/node-v20.10.0-linux-arm64/bin/npx /usr/bin/npx

```

## 4.2 安装pnpm

```bash
# 设置 npm 镜像源
npm config set registry https://registry.npmmirror.com
# 安装pnpm
npm install -g pnpm
# 设置环境变量
echo "export PATH=\$PATH:/usr/local/lib/nodejs/node-v20.10.0-linux-arm64/bin" >> ~/.bashrc
source ~/.bashrc

```

## 4.3 下载项目
```bash
git clone https://github.com/Chanzhaoyu/chatgpt-web.git
cd chatgpt-web

echo "shamefully-hoist = true"  >> .npmrc
echo "scripts-prepend-node-path = true"  >> .npmrc
echo "registry=https://registry.npmmirror.com/"  >> .npmrc

# 安装子项目依赖
cd service
pnpm install

# 安装依赖
cd ..
pnpm bootstrap

```

## 4.4 启动项目

```bash
mv ./service/.env.example ./service/.env

```
修改 ./service/.env

```
OPENAI_API_KEY=apikey
OPENAI_API_BASE_URL=http://127.0.0.1:2025
OPENAI_API_MODEL=qwen2
```

修改 ./vite.config.ts

```
port: 65504, # 端口号, 需要能外网访问
```

### 4.4.1 启动开发环境

注意: 使用开发环境启动, 容器窗口不能关闭, 关闭则失效

```bash
# 启动服务端
cd service
pnpm start

# 启动web端, 需要重新起一个容器
cd ..
pnpm dev
```

### 4.4.2 启动生产环境

#### 4.4.2.1 安装pm2

```bash
npm install -g pm2
```

#### 4.4.2.2 构建服务

修改 vite.config.ts, 在server下面添加preview配置

```
server: {
  host: '0.0.0.0',
  port: 65504,
  open: false,
  proxy: {
    '/api': {
      target: viteEnv.VITE_APP_API_BASE_URL,
      changeOrigin: true, // 允许跨域
      rewrite: path => path.replace('/api/', '/'),
    },
  },
},
preview: {
  host: '0.0.0.0',
  port: 65504,
  open: false,
  proxy: {
    '/api': {
      target: viteEnv.VITE_APP_API_BASE_URL,
      changeOrigin: true, // 允许跨域
      rewrite: path => path.replace('/api/', '/'),
    },
  },
},
```

```bash
cd /home/work/chatgpt-web/service
pnpm build

cd ..
pnpm build-only
```

#### 4.4.2.3 启动服务
```bash
cd /home/work/chatgpt-web/service
pm2 start pnpm --name service -- prod

cd ..
pm2 start pnpm --name frontend -- preview
#页面监控
pm2 list
#取消服务
pm2 stop all
pm2 delete all
```


========== mindie 录制视频步骤 ===========

配置 mindie config
启动 mindie

配置 chatweb config
启动 chatweb service
启动 chatweb client

npu-smi info
curl https://ifconfig.me

打开web, 测试问题/回答