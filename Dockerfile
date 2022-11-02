FROM python:3.7.1
ENV STATIC_URL /static
ENV STATIC_PATH /var/www/app/static
COPY . /app
WORKDIR /app
# Use an official Python runtime as a parent image
# Set the working directory to /app
COPY requirements.txt /usr/src/app/
# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
RUN pip install numpy
RUN pip install -U pip wheel cmake
RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.10.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
RUN pip install opencv-python
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install dlib
EXPOSE 5000
# Run app.py when the container launches

CMD ["python", "app.py"]