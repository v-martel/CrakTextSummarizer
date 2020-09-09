FROM continuumio/anaconda3

RUN apt-get update || : && apt-get install nodejs -y
RUN apt-get install npm -y
RUN npm i -g nodemon forever

EXPOSE 3000

ADD . /model/CrakTextSummarizer
WORKDIR /model/CrakTextSummarizer
VOLUME /model/CrakTextSummarizer

RUN /bin/bash -c "conda env create -f text_summarizer.yml"
RUN /bin/bash -c "activate text_summarizer"

ENV PYTHONPATH /model/CrakTextSummarizer

RUN echo '------------------------------- $PYTHONPATH'
