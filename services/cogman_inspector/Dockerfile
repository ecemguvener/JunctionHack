FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir "babyros==0.1.10" "pillow>=10.1" "numpy>=2.0,<3"
COPY cogman_node.py /app/cogman_node.py
CMD ["python3", "-u", "/app/cogman_node.py"]
