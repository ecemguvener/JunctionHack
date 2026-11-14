FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir "babyros==0.1.10" "websockets>=14.0" "numpy>=2.0,<3"
COPY armd_bridge_node.py /app/armd_bridge_node.py
CMD ["python3", "-u", "/app/armd_bridge_node.py"]
