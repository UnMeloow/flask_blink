#!/bin/bash
app="docker.test"
mkdir "uploads"
sudo docker ps
sudo docker run -d -p 56733:80 \
  --name=${app} \
  -v "$PWD":/app ${app}
