FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY yaml_method.py .

RUN pip install --no-cache-dir \
    torch \
    torchvision \
    tqdm

ENTRYPOINT ["python", "yaml_method.py"]
