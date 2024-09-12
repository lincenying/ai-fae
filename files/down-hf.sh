if [ $# -gt 0 ]; then
    echo "从命令行传递的参数是: $1"
    echo "从命令行传递的参数是: $2"
    cd /home/huangming/models/openmind

    # 先在openmind中创建仓库
    echo '开始克隆仓库...'
    git clone https://modelers.cn/HangZhou_Ascend/$1.git
    echo '克隆完成...'

    if [ -d "/home/huangming/models/openmind/$1" ]; then

        cd /home/huangming/models/openmind/$1
        git config user.email "lincenying@qq.com"
        git config user.name "LinCenYing"
        cd /home/huangming/models/openmind
        # 删除仓库默认描述文件, 下载模型时会从魔搭下载
        rm -rf $1/README.md

        echo '开始下载模型...'
        huggingface-cli download $2 --local-dir ./$1
    else
        echo "目录 $1 不存在。"
    fi
else
    echo "没有从命令行传递参数"
fi