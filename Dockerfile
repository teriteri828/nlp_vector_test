FROM ubuntu:18.04

ENV PYTHON_VERSION 3.7.7
ENV HOME /root
ENV PYTHON_ROOT $HOME/local/python-$PYTHON_VERSION
ENV PATH $PYTHON_ROOT/bin:$PATH
ENV PYENV_ROOT $HOME/.pyenv

RUN apt-get update -y 
RUN apt-get install tzdata -y

RUN apt-get upgrade -y
RUN  apt-get install -y \
    git \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
 && git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT \
 && $PYENV_ROOT/plugins/python-build/install.sh \
 && /usr/local/bin/python-build -v $PYTHON_VERSION $PYTHON_ROOT \
 && rm -rf $PYENV_ROOT

RUN apt-get install mecab -y
RUN apt-get install mecab-ipadic -y
RUN apt-get install libmecab-dev -y
RUN apt-get install mecab-ipadic-utf8 -y
RUN apt-get install swig -y


ARG project_dir=/web/flask/

ADD requirements.txt $project_dir
ADD application_controller.py $project_dir
ADD templates $project_dir/templates
ADD model $project_dir/model

WORKDIR $project_dir

#RUN pip install --upgrade pip
RUN pip install -r requirements.txt

#CMD ["python", "application_controller.py"]