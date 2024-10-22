# MineIE 使用指导

# 1 注册镜像
swr.cn-east-292.mygaoxinai.com/huqs/torch2.1_cann8.0.rc2_mindie1.0.rc2_py3.10_euler2.8.3_64gb_910b:v1

# 2 下载模型权重
MindIE支持.safetensors和.bin的模型权重，可以从模型网站(hf-mirror/modelscope)上获取预训练权重，或手动训练后进行转换
权重下载后需要在config中指定模型的数据类型
config.json末尾修改  
```json
  "vocab_size": 152064,    //添加逗号
  "torch_dtype": "float16" //指定数据类型 目前不支持bfloat16 请修改为float16
}
```

# 3 配置MindIE
MindIE工作目录位于 `/home/ma-user/Ascend/mindie/latest/mindie-service` 下。
想要启动服务，需要修改配置文件
```bash
vi conf/config.json
```

```json
OtherParam:
    ServeParam:
        httpsEnabled: false, //建议修改为false；如果配置为true，即开启https服务，要把服务器证书、CA证书、和服务器私钥等认证需要的文件，放置在对应的目录
    ModelDeployParam:
        npuDeviceIds: [[0,1,2,3]], //启用那几张卡，如需要8卡并行推理，修改为 [[0,1,2,3,4,5,6,7]]
        ModelParam:
            modelName: llama_65b, //可不修改
            "modelWeightPath" : "/home/ma-user/work/Qwen/Qwen_72b" //模型权重路径，修改为下载的权重路径
            "worldSize" : 4, //加载卡的数量，如修改为八卡并行推理，需要改为8
```

# 4 启动mindie

./bin/mindieservice_daemon #执行此条指令即可启动服务

# 5 测试
以千问模型为例，新建一个终端链接，复制指令
```bash
curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{
  "inputs": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>你是谁？<|im_end|>\n<|im_start|>assistant",
  "parameters": {
    "best_of": 1,
    "decoder_input_details": false,
    "details": false,
    "do_sample": true,
    "max_new_tokens": 300,
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