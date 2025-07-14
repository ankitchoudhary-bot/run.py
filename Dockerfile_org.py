FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY run.py .
COPY objective_fn.py .

RUN pip install --no-cache-dir \
    kubeflow-katib \
    torch \
    torchvision \
    tqdm \
    kubernetes

ENTRYPOINT ["python", "run.py"]
