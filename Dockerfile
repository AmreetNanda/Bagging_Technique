# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Copy current directory contents into container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port that Flask will run on
EXPOSE 5000

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production

# Run Flask when the container launches
CMD ["flask", "run"]
