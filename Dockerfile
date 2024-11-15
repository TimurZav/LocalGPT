# Базовый образ с поддержкой Python и CUDA
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Переменные среды
ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_ARGS="-DLLAMA_CUBLAS=ON" \
    FORCE_CMAKE=1 \
    SETUPTOOLS_USE_DISTUTILS=stdlib \
    OLLAMA_MODELS=/data/models \
    PYVER="3.11"

# Установить обновления, Python и основные зависимости
RUN apt update -y && apt install -y \
    libreoffice \
    python3-pip \
    software-properties-common \
    git \
    curl \
    build-essential \
    python3-dev \
    nvidia-driver-535 && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install -y python$PYVER && \
    update-alternatives --install /usr/bin/python3 python /usr/bin/python$PYVER 1 && \
    python3 -m pip install --upgrade pip

# Установка Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Скопировать зависимости и установить их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Настройка рабочей директории
WORKDIR /scripts
RUN mkdir chroma

# Запуск Ollama и основного скрипта
ENTRYPOINT ["sh", "-c", "ollama serve & python3 -u main.py"]