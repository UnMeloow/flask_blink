#!/bin/bash
app="flask_server"
sudo docker build -t ${app} .
sudo docker run -it -d -p 56733:50 \
  --name=${app} \
  -v "$PWD":/app ${app}
