# Базовый образ с поддержкой Python и CUDA
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Переменные среды
ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_ARGS="-DLLAMA_CUBLAS=ON" \
    FORCE_CMAKE=1 \
    SETUPTOOLS_USE_DISTUTILS=stdlib \
    OLLAMA_HOST="host.docker.internal:11434" \
    OLLAMA_MODELS=/data/models \
    PYVER="3.11"

# Установить обновления, Python и основные зависимости
RUN apt update -y &&  \
    apt upgrade -y &&  \
    apt install -y \
    libreoffice \
    pip \
    curl \
    nvidia-driver-535 \
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install -y python$PYVER \
    python3-pip \
    git-all && \
    apt upgrade -y && \
    apt install -y \
    build-essential \
    python3-dev && \
    python3 -m pip install --upgrade pip && \
    update-alternatives --install /usr/bin/python3 python /usr/bin/python$PYVER 1 && \
    update-alternatives --set python /usr/bin/python$PYVER && \
    python3 -V

# Скопировать зависимости и установить их
COPY requirements.txt .
RUN pip install --no-deps --no-cache-dir -r requirements.txt

# Настройка рабочей директории
WORKDIR /scripts
RUN mkdir chroma

# Запуск Ollama и основного скрипта
ENTRYPOINT ["sh", "-c", "python3 -u main.py"]
