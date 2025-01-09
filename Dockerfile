FROM ubuntu:latest
MAINTAINER Aleksei Rodionov
LABEL version="1.0"
RUN apt-get update && \
	apt-get install -y python3
COPY test.py /home/test.py
WORKDIR /home
ENTRYPOINT ["/usr/bin/python3", "test.py"]