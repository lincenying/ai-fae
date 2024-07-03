import mindspore as ms
from mindformers import AutoConfig, AutoModel, AutoTokenizer, AutoProcessor

# 指定图模式，指定使用训练卡id
ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)

# 以下两种tokenizer实例化方式选其一即可
# 1. 在线加载方式
# tokenizer = AutoTokenizer.from_pretrained("glm3_6b")
# 2. 本地加载方式
tokenizer = AutoProcessor.from_pretrained("/home/ma-user/work/mindformers/research/glm3/run_glm3_6b_finetune_2k_910b.yaml").tokenizer

# 以下两种model的实例化方式选其一即可
# 1. 直接根据默认配置实例化
# model = AutoModel.from_pretrained('glm3_6b')
# 2. 自定义修改配置后实例化
config = AutoConfig.from_pretrained("glm3_6b")
config.use_past = True  # 此处修改默认配置，开启增量推理能够加速推理性能
config.seq_length = 2048  # 根据需求自定义修改其余模型配置
config.checkpoint_name_or_path = "/home/ma-user/work/mindformers/research/glm3/6b/rank_0/glm3_6b.ckpt"
model = AutoModel.from_config(config)  # 从自定义配置项中实例化模型

role = "user"

inputs_list = ["你好", "请介绍一下华为"]

for input_item in inputs_list:
    history = []
    inputs = tokenizer.build_chat_input(input_item, history=history, role=role)
    inputs = inputs["input_ids"]
    # 首次调用model.generate()进行推理将包含图编译时间，推理性能显示不准确，多次重复调用以获取准确的推理性能
    outputs = model.generate(inputs, do_sample=False, top_k=1, max_length=config.seq_length)
    response = tokenizer.decode(outputs)
    for i, output in enumerate(outputs):
        output = output[len(inputs[i]) :]
        response = tokenizer.decode(output)
        print(response)
