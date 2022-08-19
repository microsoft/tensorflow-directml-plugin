FROM ubuntu

# Install python
RUN apt update
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install python3.7 -y
RUN ln -s /usr/bin/python3.7 /usr/bin/python