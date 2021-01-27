FROM tensorflow/tensorflow:1.15.0-gpu-py3

RUN pip install --upgrade pip
RUN pip install torch torchvision
RUN pip install tensorboardX
RUN pip install scipy
RUN pip install requests
RUN pip install Pillow
RUN pip install opencv-python
RUN pip install opencv-contrib-python
RUN pip install scikit-learn
RUN pip install scikit-image
RUN pip install matplotlib
RUN pip install numpy
RUN pip install numba
RUN pip install pandas
RUN pip install tqdm
RUN pip install progressbar
RUN pip install mtcnn
RUN pip install facenet-pytorch
RUN pip install nltk
RUN pip install pickle-mixin
RUN pip install regex
RUN pip install pyparsing
RUN pip install boto3
RUN pip install botocore
RUN pip install certifi
RUN pip install chardet
RUN pip install Click
RUN pip install cycler
RUN pip install docutils
RUN pip install idna
RUN pip install jmespath
RUN pip install joblib
RUN pip install kiwisolver
RUN pip install python-dateutil
RUN pip install transformers
RUN pip install pytz
RUN pip install s3transfer
RUN pip install sacremoses
RUN pip install sentencepiece
RUN pip install six
RUN pip install urllib3
RUN pip install langdetect
RUN pip install pytorch-transformers
RUN pip install coloredlogs
RUN pip install colorama
RUN pip install deep-translator
