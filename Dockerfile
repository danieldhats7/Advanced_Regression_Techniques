# Base image
FROM python:3.9

# Set working directory
WORKDIR /application

# Copy files
COPY application.py /application
COPY requirements.txt /application
COPY models /application/models
COPY microservices /application/microservices

# Install dependencies
RUN pip install -r requirements.txt

# Run the application
EXPOSE 8000
ENTRYPOINT FLASK_APP=/application/application.py flask run --host=0.0.0.0
