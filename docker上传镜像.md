```bash
# 获取登录指令
docker login -u xxx -p xxx swr.cn-east-292.mygaoxinai.com

# 为镜像打标签
docker tag [镜像名称1]:[版本名称1] [镜像仓库地址]/[组织名称]/[镜像名称2]:版本名称2
# 例:
docker tag mindie_server_rc2_tr5_chat_web:latest swr.cn-east-292.mygaoxinai.com/huqs/mindie_server_rc2_tr5_chat_web:latest

# 上传镜像至镜像仓库
docker push [镜像仓库地址]/[组织名称]/[镜像名称2:版本名称2]
# 例:
docker push swr.cn-east-292.mygaoxinai.com/huqs/mindie_server_rc2_tr5_chat_web:latest