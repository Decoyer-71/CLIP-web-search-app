# 1. 베이스 이미지 설정
# NVIDIA의 공식 CUDA 11.8 + Python 3.10 이미지를 사용합니다.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 시스템 의존성 설치 (OpenCV 등 C++ 라이브러리)
# Python 3.10 및 pip를 함께 설치합니다.
# Numpy, Scipy, Pillow, lxml 등 소스 컴파일에 필요한 모든 빌드 도구와 라이브러리를 설치합니다. (중복 제거 및 정리)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # --- 필수 빌드 도구 ---
    build-essential \
    gfortran \
    cmake \
    # --- Python ---
    python3.10-dev \
    python3.10 \
    python3-pip \
    python3-venv \
    # --- 수학/과학 라이브러리 의존성 (Numpy, Scipy) ---
    libopenblas-dev \
    liblapack-dev \
    # --- 이미지/비디오 라이브러리 의존성 (OpenCV, Pillow) ---
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libjpeg-dev \
    zlib1g-dev \
    # --- 기타 라이브러리 의존성 ---
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. PyTorch 및 Torchvision 설치 (CUDA 11.8 버전)
RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118

# 5. requirements.txt 복사 및 나머지 라이브러리 설치
# Docker 캐싱 효율을 위해 빌드가 오래 걸리는 numpy, scipy를 먼저 설치합니다.
COPY requirements.txt .
RUN pip install --no-cache-dir numpy==2.0.0 scipy==1.15.3
RUN pip install --no-cache-dir -r requirements.txt

# 6. 프로젝트 파일 복사
COPY . .

# 7. 앱 실행
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]