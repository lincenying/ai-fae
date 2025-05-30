
# 获取代码仓并编译
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
```

# 升级gcc版本
https://blog.csdn.net/imxiaoqy/article/details/120131541 


```bash
yum install m4 -y
wget -P '/usr/local/src' https://ftp.gnu.org/gnu/gcc/gcc-14.2.0/gcc-14.2.0.tar.gz \
&& cd /usr/local/src \
&& tar -zxvf gcc-14.2.0.tar.gz -C '/usr/local/src' \
&& cd gcc-14.2.0

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib:/usr/lib64:/usr/local/lib:/usr/local/lib64' >> /etc/profile
source /etc/profile

echo '/usr/lib' >> /etc/ld.so.conf
echo '/usr/lib64' >> /etc/ld.so.conf
echo '/usr/local/lib' >> /etc/ld.so.conf
echo '/usr/local/lib64' >> /etc/ld.so.conf
ldconfig -v

./contrib/download_prerequisites

```



```bash
cd /usr/local/src/gcc-14.2.0/gmp-6.2.1
./configure --prefix=/usr/local/gmp-6.2.1
make && make install

cd /usr/local/src/gcc-14.2.0/mpfr-4.1.0
./configure --prefix=/usr/local/mpfr-4.1.0 --with-gmp=/usr/local/gmp-6.2.1
make && make install

cd /usr/local/src/gcc-14.2.0/mpc-1.2.1
./configure --prefix=/usr/local/mpc-1.2.1 --with-gmp=/usr/local/gmp-6.2.1 --with-mpfr=/usr/local/mpfr-4.1.0
make && make install

cd /usr/local/src/gcc-14.2.0/isl-0.24
./configure --prefix=/usr/local/isl-0.24  --with-gmp-prefix=/usr/local/gmp-6.2.1
make && make install

cd ..
mkdir build
cd build
../configure  -prefix=/usr/local/gcc-12.2.0 --enable-checking=release --enable-languages=c,c++ --disable-multilib
make
make install

# 根据make install的日志结尾确定安装的目标位置，创建软链接
# 需要删除
ln -s /usr/local/gcc-12.2.0/bin/gcc /usr/bin/gcc
ln -s /usr/local/gcc-12.2.0/bin/g++ /usr/bin/g++
ln -s /usr/local/gcc-12.2.0/bin/gcc-ar /usr/bin/gcc-ar
ln -s /usr/local/gcc-12.2.0/bin/gcc-nm /usr/bin/gcc-nm
ln -s /usr/local/gcc-12.2.0/bin/gcov /usr/bin/gcov
ln -s /usr/local/gcc-12.2.0/bin/gcc-ranlib /usr/bin/gcc-ranlib

```