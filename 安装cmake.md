[当前文档访问路径](https://ai-fae.readthedocs.io/zh-cn/latest/安装cmake.html)

# 1 获取镜像  
以`swr.cn-east-292.mygaoxinai.com/huqs/pytorch2_1_0-cann7_0-py39_910b:v2`为例  
```bash
docker pull swr.cn-east-292.mygaoxinai.com/huqs/pytorch2_1_0-cann7_0-py39_910b:v2
```  

# 2 创建容器并进入  
```bash
docker run -it -e ASCEND_VISIBLE_DEVICES=3 -v /home/download/work:/home/ma-user/work/ --name test02 swr.cn-east-292.mygaoxinai.com/huqs/pytorch2_1_0-cann7_0-py39_910b:v2  
cd work
```  

# 3 安装依赖zlib  
```bash
yum install -y zlib-devel
```

# 4 安装依赖OpenSSL  
`https://www.openssl.org/source/` 获取最新的下载链接，以3.3.1为例  

```bash
cd /home/ma-user/work
wget https://www.openssl.org/source/openssl-3.3.2.tar.gz  
tar -zxvf openssl-3.3.2.tar.gz  
cd openssl-3.3.2 
./config shared zlib  --prefix=/home/ma-user/local/openssl-3.3.2 
make -j 192 
make install  
make depend  
ln -s /home/ma-user/local/openssl-3.3.2  /home/ma-user/local/openssl 
echo 'export C_INCLUDE_PATH=/home/ma-user/local/openssl/include:$C_INCLUDE_PATH' >> /home/ma-user/.bashrc
echo 'export CPLUS_INCLUDE_PATH=/home/ma-user/local/openssl/include:$CPLUS_INCLUDE_PATH' >> /home/ma-user/.bashrc
echo 'export LD_LIBRARY_PATH=/home/ma-user/local/openssl/lib:$LD_LIBRARY_PATH' >> /home/ma-user/.bashrc

```  

# 5 修改配置文件  
```bash
echo '/home/ma-user/local/openssl/lib' >> /etc/ld.so.conf
ldconfig  
echo 'export OPENSSL_ROOT_DIR=/home/ma-user/local/openssl' >> /home/ma-user/.bashrc
ln -s /home/ma-user/local/openssl/bin/openssl  /home/ma-user/bin/openssl
ln -s /home/ma-user/local/openssl/bin/c_rehash /home/ma-user/bin/c_rehash
source ~/.bashrc
```  

# 6 安装cmake  
`https://cmake.org/download/` 获取最新的下载链接，以3.30.0为例  

```bash
cd /home/ma-user/work  
wget https://github.com/Kitware/CMake/releases/download/v3.30.0/cmake-3.30.0.tar.gz  
tar -zxvf cmake*
cd cmake-3.30.0
./bootstrap  --prefix=/home/ma-user/local/cmake-3.31.0.rc1
make -j 192
make install
ln -s /home/ma-user/local/cmake/bin/cmake /home/ma-user/bin/cmake
ln -s /home/ma-user/local/cmake/bin/ctest /home/ma-user/bin/ctest
ln -s /home/ma-user/local/cmake/bin/cpack /home/ma-user/bin/cpack
# 确认版本  
cmake --version
```