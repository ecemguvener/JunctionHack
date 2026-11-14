FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir "openvino==2026.4.1" "babyros==0.1.10" "scikit-learn>=1.5" "scipy>=1.14" "opencv-python-headless>=4.10"
COPY perception_policy_node.py /app/perception_policy_node.py
CMD ["python3", "-u", "/app/perception_policy_node.py"]
