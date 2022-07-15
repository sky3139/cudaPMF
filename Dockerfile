FROM nvidia/cuda:11.0-devel
ARG DEBIAN_FRONTEND=noninteractive
RUN uname -v
RUN sed -i s:/archive.ubuntu.com:/mirrors.tuna.tsinghua.edu.cn/ubuntu:g /etc/apt/sources.list
RUN cat /etc/apt/sources.list
# RUN apt-get clean
RUN rm /etc/apt/sources.list.d/cuda.list
# RUN apt-get update && apt-get install -y apt-transport-https
RUN apt-get -y update
# COPY ./keyboard /etc /default /keyboard
RUN apt install cmake gcc g++ sudo -y
RUN apt install libpcl-dev -y
RUN sudo apt install -y libopencv-dev libomp-dev git 
ADD run.sh /
# ADD  ./ /root/plane/
# RUN rm -rf /root/plane/build/*