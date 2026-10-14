# Sandbox image for generated experiment code. Built once by a human
# (`python -m research build-sandbox`); experiments run with --network none,
# so every library the coder may use must be baked in here.
FROM python:3.12.7-slim
RUN pip install --no-cache-dir \
        numpy==2.1.3 scipy==1.14.1 scikit-learn==1.5.2 pandas==2.2.3
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /tmp
USER 65534:65534
