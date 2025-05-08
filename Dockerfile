FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y python3 python3-pip git cron && \
    rm -rf /var/lib/apt/lists/*

# Pip upgrade & essentials
RUN pip3 install --upgrade pip
RUN pip3 install vllm transformers datasets peft bitsandbytes accelerate wandb jupyterlab notebook huggingface_hub pynvml

# JupyterLab Git integration
RUN pip3 install jupyterlab-git
RUN jupyter labextension install @jupyterlab/git

# Create work directory
RUN mkdir /app
WORKDIR /app

# Expose ports (JupyterLab + REST API)
EXPOSE 8000 8888

# Environment variables (can be filled dynamically later)
ENV MODEL_NAME="/app/models/default-model"
ENV WANDB_API_KEY=""
ENV WANDB_MODE="online"
ENV HUGGINGFACE_TOKEN=""

# Add cron job for GPU monitoring
COPY check_gpu_usage.sh /app/check_gpu_usage.sh
RUN chmod +x /app/check_gpu_usage.sh
RUN echo "* * * * * root /app/check_gpu_usage.sh" > /etc/cron.d/gpucheck
RUN chmod 0644 /etc/cron.d/gpucheck && crontab /etc/cron.d/gpucheck

# Start JupyterLab and vLLM REST API
CMD ["bash", "-c", "\
    service cron start && \
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root & \
    python3 -m vllm.entrypoints.api_server --model ${MODEL_NAME} --host 0.0.0.0 --port 8000 --dtype auto \
    "]
