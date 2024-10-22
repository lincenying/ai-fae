```bash
wget http://mirrors.ustc.edu.cn/gnu/make/make-4.3.tar.gz
tar xf make-4.3.tar.gz 
cd make-4.3/
# 安装到指定目录
./configure  --prefix=/home/ma-user/local/make-4.3
make
make install
ln -s /home/ma-user/local/make-4.3 /home/ma-user/local/make
ln -s /home/ma-user/local/make/bin/make /home/ma-user/bin/make
hash -r
make -v
# make -v 版本不对是因为缓存的原因，断开重连或执行 hash -r 刷新缓存即可
```