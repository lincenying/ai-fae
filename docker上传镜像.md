[当前文档访问路径](https://ai-fae.readthedocs.io/zh-cn/latest/docker上传镜像.html)

# 上传镜像至SWR

```bash
# 获取登录指令
docker login -u xxx -p xxx swr.cn-east-292.mygaoxinai.com

# 将容器保存成镜像
docker commit mindie_server_hm mindie_server_chatgpt_web_910b:20240923_T65

# 为镜像打标签
docker tag [镜像名称1]:[版本名称1] [镜像仓库地址]/[组织名称]/[镜像名称2]:版本名称2
# 例:
docker tag mindie_server_chatgpt_web_910b:20240923_T65 swr.cn-east-292.mygaoxinai.com/huqs/mindie_server_chatgpt_web_910b:20240923_T65

# 上传镜像至镜像仓库
docker push [镜像仓库地址]/[组织名称]/[镜像名称2:版本名称2]
# 例:
docker push swr.cn-east-292.mygaoxinai.com/huqs/mindie_server_chatgpt_web_910b:20240923_T65


# 下载镜像到本地
```bash
docker save mindie_server_chatgpt_web_910b > /opt/data/docker_images/mindie_server_chatgpt_web_910b.tar

obsutil cp /opt/data/docker_images/mindie_server_chatgpt_web_910b.tar obs://docker_images/
```