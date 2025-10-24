# ===============================
# CUDA対応 + Python 3.10 + CPU fallback
# ===============================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Python 3.10 と pip をインストール
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    cmake \
    ffmpeg \
    build-essential \
    python3.10-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# 作業ディレクトリ
WORKDIR /tranosuke

# requirements.txt をコピー（親ディレクトリから）
COPY ../requirements.txt .

# Pythonライブラリをインストール
RUN pip --default-timeout=10000 install --no-cache-dir -r requirements.txt
RUN pip --default-timeout=10000 install git+https://github.com/DwangoMediaVillage/pydomino

# アプリ本体をコピー
COPY ../ .

# Streamlit 設定（外部アクセスを許可）
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# アプリ実行（Flask例）
CMD ["python", "app.py"]