[当前文档访问路径](https://ai-fae.readthedocs.io/zh-cn/latest/mindie-benchmark使用指导.html)

# 1. BenchMark client模式
client模式，需要安装MindIE-client，使用时需要开启服务！！
进入到Server的目录下，启动服务。

```bash
cd /path/mindie-service/
./bin/mindieservice_daemon
```

打开另一个窗口，运行benchmark命令即可。

## 1.1	BenchMark client测性能：
### 1.1.1	按并发数测试：
#### 1）带后处理，使用文本流式推理模式

```bash
export SMPL_PARAM="{\"temperature\":0.5,\"top_k\":10,\"top_p\":0.9,\"typical_p\":0.9,\"seed\":1234,\"repetition_penalty\":1,\"watermark\":true,\"truncate\":10}"  
benchmark \
--DatasetPath "/{模型库安装路径}/atb-models/tests/modeltest/dataset/full/GSM8K" \
--DatasetType "gsm8k" \
--ModelName llama_7b \
--ModelPath "/{data}/llama_7b" \
--TestType client \
--Http "http://{ipAddress}:{port}" \
--Concurrency 128 \
--TaskKind stream \
--Tokenizer True \
--MaxOutputLen 512 \
--DoSampling True \
--SamplingParams $SMPL_PARAM
```

注释：
--TestType client benchmark的模式，此处为client模式
--Http 调用的接口和端口，与server中config.json的ipAddress和port保持一致
--TaskKind stream Client不同推理模式，此处为文本流式推理
 
#### 2）不带后处理，使用文本流式推理模式

```bash
benchmark \
--DatasetPath "/{模型库安装路径}/atb-models/tests/modeltest/dataset/full/GSM8K" \
--DatasetType "gsm8k" \
--ModelName llama_7b \
--ModelPath "/{data}/llama_7b" \
--TestType client \
--Http "http://{ipAddress}:{port}" \
--Tokenizer True \
--Concurrency 128 \
--TaskKind stream \
--MaxOutputLen 512
```

#### 3）不带后处理，使用文本非流式推理模式

```bash
benchmark \
--DatasetPath "/{模型库安装路径}/atb-models/tests/modeltest/dataset/full/GSM8K" \
--DatasetType "gsm8k" \
--ModelName llama_7b \
--ModelPath "/{data}/llama_7b" \
--TestType client \
--Http "http://{ipAddress}:{port}" \
--Concurrency 128 \
--TaskKind text \
--Tokenizer True \
--MaxOutputLen 512
```

注释：
--TaskKind text client对应的不同模式，此处为非流式推理模式
 

### 1.1.2	按频率测试：
#### 1）按均一分布发送

```bash
benchmark \
--DatasetPath "/{模型库安装路径}/atb-models/tests/modeltest/dataset/full/GSM8K" \
--DatasetType "gsm8k" \
--ModelName llama_7b \
--ModelPath "/{data}/llama_7b" \
--TestType client \
--Http "http://{ipAddress}:{port}" \
--Concurrency 128 \
--TaskKind stream \
--MaxOutputLen 512 \
--RequestRate 2,4 \
--Tokenizer True \
--Distribution uniform
```

注释：
--RequestRate 指定一组发送频率，按照Distribution参数设置的模式进行发送，以每个频率完成一次推理
--Distribution 请求发送模式

#### 2）按泊松分布发送

```bash
benchmark \
--DatasetPath "/{模型库安装路径}/atb-models/tests/modeltest/dataset/full/GSM8K" \
--DatasetType "gsm8k" \
--ModelName llama_7b \
--ModelPath "/{data}/llama_7b" \
--TestType client \
--Http "http://{ipAddress}:{port}" \
--Concurrency 128 \
--TaskKind stream \
--MaxOutputLen 512 \
--RequestRate 2,4 \
--Tokenizer True \
--Distribution poisson
```

## 1.2 测精度：
并发数--Concurrency需设置为1，确保模型推理时是1batch输入，这样才可以和纯模型比对精度。
使用CEval比对精度时，MaxOutputLen应该设为20，MindIE-Server的config.json文件中MaxSeqlen需要设置为3072。
使用MMLU比对精度时，MaxOutputLen应该设为20，MindIE-Server的config.json文件中MaxSeqlen需要设置为3600，该数据集中有约为1.4w条数据，推理耗时会比较长。
目前对外呈现，我们可以测试CEval 5-shot、MMLU 5-shot、gsm8k的精度。
注意事项，在source完所有环境变量之后，需要开启确定性计算环境变量！！！

```bash
export LCCL_DETERMINISTIC=1
export HCCL_DETERMINISTIC=1
export ATB_MATMUL_SHUFFLE_K_ENABLE=0
export ATB_LLM_LCOC_ENABLE=0

benchmark \ 
--DatasetPath "/{模型库安装路径}/atb-models/tests/modeltest/dataset/full/CEval" \
--DatasetType "ceval" \
--ModelName "llama2_7b" \
--ModelPath "/${home}/llama2_7b" \
--TestType client \
--Http "http://{ipAddress}:{port}" \
--Concurrency 1 \
--MaxOutputLen 20 \
--Tokenizer True \
--TestAccuracy True
```

注释：
--TestAccuracy测试精度标识

# 2. BenchMark engine模式
一般用OA和gms8k模式
engine模式直接调用MindIE-Server的python北向接口，运行时不开启服务。

## 2.1	Benchmark engine测性能：
### 2.1.1 不带后处理，使用文本模式测。

```bash
benchmark \
--DatasetPath "/{模型库安装路径}/atb-models/tests/modeltest/dataset/full/GSM8K" \
--DatasetType "gsm8k" \
--ModelName "baichuan2_13b" \
--ModelPath "/${home}/baichuan2-13b" \
--TestType engine \
--Concurrency 50 \
--Tokenizer True \
--MaxOutputLen 512 \
--WarmupSize 10
```

注释：
--DatasetPath 为数据集路径
--DatasetType 数据集类型
--ModelName 模型权重目录，需要与Server的config.json保持一致
--ModelPath 模型名称，需要与Server的config.json保持一致
--TestType 测试模式，此处为engine
--Concurrency 并发数
--MaxOutputLen 512 最大输出长度
--WarmupSize 为warm up的条数，默认为10

### 2.1.2 带后处理，使用文本模式测。

```bash
export SMPL_PARAM="{\"temperature\":0.5,\"top_k\":10,\"top_p\":0.9,\"typical_p\":0.9,\"seed\":1234,\"repetition_penalty\":1,\"watermark\":true,\"truncate\":10}"  
benchmark \
--DatasetPath "/{模型库安装路径}/atb-models/tests/modeltest/dataset/full/GSM8K" \
--DatasetType "gsm8k" \
--ModelName "baichuan2_13b" \
--ModelPath "/${home}/baichuan2-13b" \
--TestType engine \
--Concurrency 50 \
--MaxOutputLen 512 \
--Tokenizer True \
--DoSampling True \
--SamplingParams=$SMPL_PARAM
```

注释：
--DoSampling 输出结果采样标识
--SamplingParams 输出结果采样标识为True时有效

## 2.2 Benchmark engine测精度：
注意事项，在source完所有环境变量之后，需要开启确定性计算环境变量！！！

```bash
export LCCL_DETERMINISTIC=1
export HCCL_DETERMINISTIC=1
export ATB_MATMUL_SHUFFLE_K_ENABLE=0
export ATB_LLM_LCOC_ENABLE=0

benchmark \
--DatasetPath "/{模型库安装路径}/atb-models/tests/modeltest/dataset/full/CEval" \
--DatasetType "ceval" \
--ModelName "llama2_7b" \
--ModelPath "/${home}/llama2_7b" \
--TestType engine \
--Concurrency 1 \
--Tokenizer True \
--MaxOutputLen 20 \
--TestAccuracy True
```

注释：
--TestAccuracy测试精度标识
boolQ和humaneval是内部数据集，无法直接使用--TestAccuracy True屏显精度。
