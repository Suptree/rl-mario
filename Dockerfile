FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

RUN apt update

RUN pip install gym==0.25.1
RUN apt install python-opengl
RUN apt install python3-tk