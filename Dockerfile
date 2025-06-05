# Use an official Python runtime as a parent image
# Choose a version compatible with your development environment (e.g., 3.11)
# Using slim-bookworm for a smaller base image
FROM python:3.11-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt ./

# Install any needed system dependencies (if PyMuPDF or others require them - often not needed with wheels)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#  && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# --no-cache-dir keeps the image size down
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Make port 8501 available to the world outside this container (Streamlit default)
EXPOSE 8501

# Define environment variable for Streamlit health check (optional but good practice)
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Run streamlit_app.py when the container launches
# Use exec form to ensure signals are handled correctly
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"] 