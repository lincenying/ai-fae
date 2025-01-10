```bash
# ↓↓↓↓↓↓↓↓↓ 安装conda 可不装 ↓↓↓↓↓↓↓↓↓

# 下载Miniconda安装脚本, 可从 https://docs.anaconda.com/miniconda/install/ 查询对应版本
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
 
# 使脚本可执行
chmod +x Miniconda3-latest-MacOSX-x86_64.sh
 
# 安装Miniconda
./Miniconda3-latest-MacOSX-x86_64.sh
 
# 初始化Conda
conda init
 
# 重启终端或者更新你的shell配置文件
source ~/.bash_profile  # 或者你使用的相应配置文件

# ↑↑↑↑↑↑↑↑ 安装conda 可不装 ↑↑↑↑↑↑↑↑

cd /Users/lincenying/langchain

# 拉取项目代码
git clone https://github.com/chatchat-space/Langchain-Chatchat.git

# 安装 Poetry
cd Langchain-Chatchat 
# 创建虚拟环境
python3 -m venv .venv
# 激活虚拟环境
. .venv/bin/activate
# 升级pip
pip install --upgrade pip

pip install -U pip setuptools
pip install poetry

# 安装 Langchain-Chatchat 依赖
cd /Users/lincenying/langchain/Langchain-Chatchat/libs/chatchat-server
poetry install --with lint,test -E xinference
pip install -e .

mkdir -p /Users/lincenying/langchain/Langchain-Chatchat/chatchat_data

export CHATCHAT_ROOT=/Users/lincenying/langchain/Langchain-Chatchat/chatchat_data/   # 设置 CHATCHAT_ROOT 环境变量

############# 另起一个窗口 ###################
# 安装 xinference
cd /Users/lincenying/langchain
mkdir xinference
cd xinference
# 创建虚拟环境
python3 -m venv .venv
# 激活虚拟环境
. .venv/bin/activate
# 升级pip
pip install --upgrade pip

pip install "xinference[transformers]" # pip install "xinference[all]" 安装所有引擎
pip install xinference-client
pip install sentence-transformers
pip install "numpy<2"
xinference-local --host 0.0.0.0 --port 9997
# 浏览器打开 http://0.0.0.0:9997/ui/#/launch_model/llm 启动对应模型
# llm_models: qwen2.5-instruct
# embed_models: bge-large-zh-v1.5

############## 切回之前窗口 ################

# 启动 Langchain-Chatchat 服务
pip install xinference-client
# 执行初始化
cd /Users/lincenying/langchain/Langchain-Chatchat/libs/chatchat-server
python chatchat/cli.py init
# chatchat init

# 修改配置文件
vi /Users/lincenying/langchain/Langchain-Chatchat/chatchat_data/model_settings.yaml
```

```yaml
DEFAULT_LLM_MODEL: qwen2.5-instruct
DEFAULT_EMBEDDING_MODEL: bge-large-zh-v1.5

MODEL_PLATFORMS:
  - platform_name: xinference
    platform_type: xinference
    llm_models:
      - qwen2.5-instruct
    embed_models:
      - bge-large-zh-v1.5
```

```bash
# 知识库
# 将知识库文件放入 /Users/lincenying/langchain/Langchain-Chatchat/chatchat_data/data/knowledge_base/samples/content/test_files 文件夹

# 安装 xlrd, 支持xlsx文件
pip install xlrd

# 初始化知识库
chatchat kb -r

# 启动服务
chatchat start -a
```

# 注意


# 遇到的问题及解决方法

:::xinference:::
对话时报`An error occurred during streaming`错误
```bash
# 降级 transformers
pip install transformers===4.41.2
```

:::xinference:::
报`Cannot import name 'EncoderDecoderCache' from 'transformers'`错误
```bash
pip install peft==0.10.0
```

报`Client.__init__() got an unexpected keyword argument 'proxies'`错误
```bash
# 降级 httpx
pip install httpx===0.27.2
```

报`NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'`警告
```bash
pip install urllib3==1.26.6
```