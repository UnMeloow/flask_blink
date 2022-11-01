FROM python:3
ENV STATIC_URL /static
ENV STATIC_PATH /var/www/app/static
COPY . /app
WORKDIR /app
# Use an official Python runtime as a parent image
# Set the working directory to /app
COPY requirements.txt /usr/src/app/
# Install any needed packages specified in requirements.txt
RUN pip install numpy
RUN pip install -U pip wheel cmake
RUN pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.0-py3-none-any.whl
RUN pip install --trusted-host pypi.python.org -r requirements.txt
# Make port 8000 available to the world outside this container
EXPOSE 8000
# Run app.py when the container launches
CMD ["python", "app.py"]
