#!/bin/bash
app="flask_server"
docker build -t ${app} .
docker run -p 5000:5000 ${app}
