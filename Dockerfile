FROM python:3.8
WORKDIR /usr/src
RUN git clone https://github.com/alexanderDuenas/remote_sensing_satellite_image.git
RUN pip install -r remote_sensing_satellite_image/requirements.txt
RUN python -m pip install -q tensorflow==2.2.1
RUN pip install -U -q segmentation-models
RUN pip install -q keras==2.5

CMD ["bash"] 

