# Используйте базовый образ с поддержкой Python
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_ARGS="-DLLAMA_CUBLAS=ON" \
    FORCE_CMAKE=1 \
    SETUPTOOLS_USE_DISTUTILS=stdlib

# Обновляем пакеты и устанавливаем libreoffice
RUN apt update -y && apt upgrade -y && apt install libreoffice -y && apt install pip -y  \
    && apt install nvidia-driver-535 -y

ARG PYVER="3.11"
# Install Python (software-properties-common), Git, and Python utilities
# Learn about the deadsnakes Personal Package Archives, hosted by Ubuntu:
# https://www.youtube.com/watch?v=Xe40amojaXE
RUN apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y python$PYVER \
    python3-pip \
    git-all

# Обновите пакеты до последней версии
RUN apt-get -y upgrade && apt-get install -y build-essential python3-dev

# Обновить PIP (менеджер пакетов Python)
RUN python3 -m pip install --upgrade pip
RUN python3 -V
# Установите PYVER в качестве интерпретатора Python по умолчанию
RUN update-alternatives --install /usr/bin/python3 python /usr/bin/python$PYVER 1
RUN update-alternatives --set python /usr/bin/python$PYVER
RUN update-alternatives --set python /usr/bin/python$PYVER
RUN python3 -V

# Копируйте файлы зависимостей (если есть) и другие необходимые файлы
COPY requirements.txt .
RUN pip install --no-deps --no-cache-dir -r requirements.txt

# Создайте директорию для приложения
RUN mkdir /scripts && mkdir /scripts/chroma
WORKDIR /scripts

# Не копируйте большие модели в образ, так как это может сделать его слишком объемным
# Вместо этого, они будут подключены через volumes в docker-compose.yml

# Запустите скрипт при запуске контейнера
CMD ["python3", "-u", "main.py"]