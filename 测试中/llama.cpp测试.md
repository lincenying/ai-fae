
# 获取代码仓并编译
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
```

# 升级gcc版本
https://blog.csdn.net/imxiaoqy/article/details/120131541 


```bash
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
../configure --enable-threads=posix --enable--long-long --enable-languages=c,c++ --disable-checking --disable-multilib
../configure --prefix=/root/Downloads/gcc-14.2.0 --enable-bootstrap --enable-long-long --enable-checking=release --enable-languages=c,c++ --disable-multilib
ar --plugin /usr/libexec/gcc/aarch64-linux-gnu/7.3.0/liblto_plugin.so rc
ar --plugin /usr/local/Ascend/ascend-toolkit/8.0.RC2/toolkit/toolchain/hcc/libexec/gcc/aarch64-target-linux-gnu/7.3.0/ rc
```