FROM continuumio/anaconda3

ADD . /model/CrakTextSummarizer
WORKDIR /model/CrakTextSummarizer
VOLUME /model/CrakTextSummarizer

RUN apt-get update || : && apt-get install nodejs -y
RUN apt-get install npm -y

RUN npm i -g nodemon forever

RUN conda env create -f text_summarizer.yml

ENV PATH /model/CrakTextSummarizer/env/bin:$PATH
ENV PYTHONPATH /model/CrakTextSummarizer:$PATH

COPY src/main.py .
CMD ["python", "conda", "nodemon", "src/main.py"]

EXPOSE 3000
