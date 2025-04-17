FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /Docker_Image
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY yolo_fruits_and_vegetables_v3.pt .
EXPOSE 8000
CMD ["python", "app.py"]
