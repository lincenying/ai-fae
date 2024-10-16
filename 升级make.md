
# 升级gcc版本
https://blog.csdn.net/CLinuxF/article/details/108705142 


```bash
wget http://mirrors.ustc.edu.cn/gnu/make/make-4.3.tar.gz
tar xf make-4.3.tar.gz 
cd make-4.3/
# 安装到指定目录
./configure  --prefix=/usr/local/make
make
make install
make -v
# 此时的 make 还是3.82 与环境变量有关系
/usr/local/make/bin/make -v
————————————————

                            版权声明：本文为博主原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接和本声明。
                        
原文链接：https://blog.csdn.net/CLinuxF/article/details/108705142

# 根据make install的日志结尾确定安装的目标位置，创建软链接
# 需要删除
rm -rf /usr/bin/gc*
rm -rf /usr/bin/g+*
ln -s /usr/local/gcc-14.2.0/bin/gcc /usr/bin/gcc
ln -s /usr/local/gcc-14.2.0/bin/g++ /usr/bin/g++
ln -s /usr/local/gcc-14.2.0/bin/gcc-ar /usr/bin/gcc-ar
ln -s /usr/local/gcc-14.2.0/bin/gcc-nm /usr/bin/gcc-nm
ln -s /usr/local/gcc-14.2.0/bin/gcov /usr/bin/gcov
ln -s /usr/local/gcc-14.2.0/bin/gcc-ranlib /usr/bin/gcc-ranlib

```