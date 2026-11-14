FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libusb-1.0-0 libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "opencv-python-headless>=4.10" "babyros==0.1.10" "numpy>=2.0,<3"
COPY camera_pub.py /app/camera_pub.py
CMD ["python3", "-u", "/app/camera_pub.py"]
