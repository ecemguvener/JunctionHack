FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libgl1 libgles2 libglvnd0 libosmesa6 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "mujoco==4.0.1" "frax[ik]==0.2.0" "babyros==0.1.10"
COPY twin_server.py /app/twin_server.py
CMD ["python3", "-u", "/app/twin_server.py"]
