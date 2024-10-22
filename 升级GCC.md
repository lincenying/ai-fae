
# 升级gcc版本
https://blog.csdn.net/imxiaoqy/article/details/120131541 


```bash
yum install m4 -y
wget https://ftp.gnu.org/gnu/gcc/gcc-14.2.0/gcc-14.2.0.tar.gz
tar -zxvf gcc-14.2.0.tar.gz
cd gcc-14.2.0

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib:/usr/lib64' >> /etc/profile
source /etc/profile

echo '/usr/lib' >> /etc/ld.so.conf
echo '/usr/lib64' >> /etc/ld.so.conf
ldconfig -v
./contrib/download_prerequisites

mkdir build
cd build
../configure  -prefix=/home/ma-user/local/gcc-14.2.0 --enable-checking=release --enable-languages=c,c++ --disable-multilib
make -j 64
make install
ln -s /home/ma-user/local/gcc-14.2.0 /home/ma-user/local/gcc
echo 'export PATH=/home/ma-user/local/gcc/lib:$PATH' >> /home/ma-user/.bashrc

ln -s /home/ma-user/local/gcc/bin/c++ /home/ma-user/bin/c++
ln -s /home/ma-user/local/gcc/bin/cpp /home/ma-user/bin/cpp
ln -s /home/ma-user/local/gcc/bin/g++ /home/ma-user/bin/g++
ln -s /home/ma-user/local/gcc/bin/gcc /home/ma-user/bin/gcc
ln -s /home/ma-user/local/gcc/bin/gcc-ar /home/ma-user/bin/gcc-ar
ln -s /home/ma-user/local/gcc/bin/gcc-nm /home/ma-user/bin/gcc-nm
ln -s /home/ma-user/local/gcc/bin/gcc-ranlib /home/ma-user/bin/gcc-ranlib
ln -s /home/ma-user/local/gcc/bin/gcov /home/ma-user/bin/gcov
ln -s /home/ma-user/local/gcc/bin/gcov-dump /home/ma-user/bin/gcov-dump
ln -s /home/ma-user/local/gcc/bin/gcov-tool /home/ma-user/bin/gcov-tool
ln -s /home/ma-user/local/gcc/bin/lto-dump /home/ma-user/bin/lto-dump

echo 'export CXX=/home/ma-user/bin/g++' >> /home/ma-user/.bashrc
echo 'export CC=/home/ma-user/bin/gcc' >> /home/ma-user/.bashrc
echo 'export LD_LIBRARY_PATH=/home/ma-user/local/gcc/lib64:$LD_LIBRARY_PATH' >> /home/ma-user/.bashrc
echo 'export LIBRARY_PATH=/home/ma-user/local/gcc/lib64:$LIBRARY_PATH' >> /home/ma-user/.bashrc
echo 'export C_INCLUDE_PATH=/home/ma-user/local/gcc/include:$C_INCLUDE_PATH' >> /home/ma-user/.bashrc
echo 'export CPLUS_INCLUDE_PATH=/home/ma-user/local/gcc/include:$CPLUS_INCLUDE_PATH' >> /home/ma-user/.bashrc
echo 'export MANPATH=/home/ma-user/local/gcc/share/man:$MANPATH' >> /home/ma-user/.bashrc
echo 'export PKG_CONFIG_PATH=/home/ma-user/local/gcc/lib64/pkgconfig:$PKG_CONFIG_PATH' >> /home/ma-user/.bashrc

``` 