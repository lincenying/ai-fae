默认情况下，Docker 在 Linux 上安装时会将其数据（如镜像和容器）存储在 /var/lib/docker 目录下，这个目录通常位于根分区 /。
如果你的根分区 / 的空间不足，你可以考虑将 Docker 的数据目录移动到其他分区，或者在安装 Docker 时直接配置它使用其他分区。

要更改 Docker 的默认存储位置，你可以通过修改 Docker 的配置文件或在启动 Docker 服务时设置环境变量来指定新的数据目录。
下面是一些步骤说明如何进行配置：

# 1. 停止 docker 服务

```bash
systemctl stop docker
# 如果没有权限, 可以使用 sudo
sudo systemctl stop docker

```

# 2. 创建新的存储目录

```bash
mkdir -p /data/docker
```

# 3. 修改 Docker 配置文件

```bash
vi /etc/docker/daemon.json
```

如果是空文件, 添加以下内容

```json
{
    "data-root": "/data/docker"
}
```

如果不是空文件, 在`}`上面一行添加以下内容

```json
    "data-root": "/data/docker"
```

# 4. 移动现有的 Docker 数据

```bash
mv /var/lib/docker/* /data/docker/
```

# 5. 重新启动 Docker 服务

```bash
systemctl start docker
```

# 6. 验证修改

```bash
docker info | grep 'Docker Root Dir'
# 打印出以下内容:
# Docker Root Dir: /data/docker
```