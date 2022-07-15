
# 将测试文件放在宿主机器 /task6 文件下
sudo mkdir -p /task6
sudo mkdir -p /task6_result/spirt
sudo rm /task6_result/spirt/*
<!-- sudo cp *.pcd /task6 -->
# build docker 构建镜像 
sudo docker build -t pl:v4 .
# 或者导入镜像
sudo docker load -i pl_v3.tar
# run 运行容器
# 代码路径 ：/home/u20/gitee/cudaPMF 执行命令前将其替换
<!-- 运行后移除的命令 -->
sudo docker run --rm -it --gpus all  -v /task6:/task6 \
-v /home/u20/d2/cudaPMF:/root/cudaPMF \
-v /task6_result/spirt:/task6_result/spirt pl:v4 bash run.sh

<!-- 运行后不移除的命令 -->
sudo docker run  -it --gpus all  -v /task6:/task6 \
-v /task6_result/spirt/cudaPMF:/root/cudaPMF \
-v /task6_result/spirt:/task6_result/spirt pl:v3 /usr/bin/bash /root/cudaPMF/run.sh