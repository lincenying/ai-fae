# 1 获取镜像  
以`swr.cn-east-292.mygaoxinai.com/huqs/pytorch2_1_0-cann7_0-py39_910b:v2`为例  
```docker pull swr.cn-east-292.mygaoxinai.com/huqs/pytorch2_1_0-cann7_0-py39_910b:v2````  

# 2 创建容器并进入  
```docker run -it -e ASCEND_VISIBLE_DEVICES=3 -v /home/download/work:/home/ma-user/work/ --name test02 swr.cn-east-292.mygaoxinai.com/huqs/pytorch2_1_0-cann7_0-py39_910b:v2  
cd work```  

# 3 安装依赖zlib  
`yum install -y zlib-devel`  

# 4 安装依赖OpenSSL  
`https://www.openssl.org/source/` 获取最新的下载链接，以3.3.1为例  
```wget https://www.openssl.org/source/openssl-3.3.1.tar.gz  
tar -zxvf openssl-3.3.1.tar.gz  
cd openssl-3.3.1.tar.gz  
./config shared zlib  --prefix=/usr/local/openssl && make && make install  
./config -t  
make depend  
cd /usr/local  
ln -s openssl ssl```  

# 5 修改配置文件  
`vim /etc/ld.so.conf`  
文件末尾添加
`/usr/local/openssl/lib`  
保存并退出
执行指令
```ldconfig  
vim etc/profile```  
文件末尾添加
```export OPENSSL=/usr/local/openssl/bin  
export PATH=$OPENSSL:$PATH:$HOME/bin  
export OPENSSL_ROOT_DIR=/usr/local/openssl```  
保存并退出
重新登录镜像以完成安装

# 6 安装cmake  
`https://cmake.org/download/` 获取最新的下载链接，以3.30.0为例  
```cd /home/ma-user/work  
wget https://github.com/Kitware/CMake/releases/download/v3.30.0/cmake-3.30.0.tar.gz  
tar -zxvf cmake-3.30.0.tar.gz  
cd cmake-3.30.0.tar.gz  
./bootstrap  
make  
make install```  

# 7 确认版本  
`cmake --version`  

# 8 打包镜像并上传  
退出镜像  
`docker commit 容器名字 镜像链接:版本号`   
```docker commit test02 swr.cn-east-292.mygaoxinai.com/huqs/pytorch2_1_0-cann7_0-py39_910b:v3 #更新到v3  
docker push swr.cn-east-292.mygaoxinai.com/huqs/pytorch2_1_0-cann7_0-py39_910b:v3```  