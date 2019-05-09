# Use anaconda python image
FROM continuumio/anaconda3

# Set working directory
WORKDIR /app

# Copy the files over to the directory
COPY . /app

# Update apt
RUN apt-get -y update

# Install dlib and its pre recs
# See: https://github.com/ageitgey/face_recognition/blob/master/Dockerfile
RUN apt-get install -y python3-setuptools

RUN apt-get install -y --fix-missing \
    build-essential \
    python3-dev \
    python3-numpy \
    cmake \
    gfortran \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN cd ~ && \
    mkdir -p dlib && \
    git clone -b 'v19.9' --single-branch https://github.com/davisking/dlib.git dlib/ && \
    cd  dlib/ && \
    python3 setup.py install --yes USE_AVX_INSTRUCTIONS

# Install packages from requiremenst
RUN pip install -r requirements.txt

# Expose the bokeh port
EXPOSE 5006

# Launch app using default headers (for SSL purposes) and enabling connections from
# anywhere
CMD ["bokeh", "serve", "myapp.py", "--use-xheaders", "--allow-websocket-origin=*"]
