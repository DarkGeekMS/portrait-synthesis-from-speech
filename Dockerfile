FROM tensorflow/tensorflow:1.15.0-gpu-py3

RUN pip install torch torchvision
RUN pip install tensorboardX
RUN pip install scipy==1.3.3
RUN pip install requests==2.22.0
RUN pip install Pillow==6.2.1
RUN pip install opencv-python
RUN pip install opencv-contrib-python
RUN pip install scikit-learn
RUN pip install scikit-image
RUN pip install tqdm
RUN pip install progressbar
RUN pip install mtcnn
RUN pip install facenet-pytorch
RUN pip install nltk
