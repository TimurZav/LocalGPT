# Llama CPP

## 📡 Description

This program is designed to provide an answer to a question based on uploaded documents using Llama-model. The program is running on ubuntu:22.04

## 📜 Installation

![alt text](https://logos-world.net/wp-content/uploads/2021/02/Docker-Symbol.png)

Use the [docker](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-compose-on-ubuntu-20-04) to run projects for this program.
```sh
sudo apt install docker-compose
```

### CPU

```sh
git clone https://github.com/agladsoft/LocalChatGPT.git

cd LocalChatGPT

sudo docker-compose up
```

Remove code in `docker-compose.yml`
```docker
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [ gpu ]
```
AND
change code in `app.py` class `Llama`. Code `n_gpu_layers=35` to `n_gpu_layers=0`

### GPU

```sh
sudo apt update && sudo apt upgrade

ubuntu-drivers devices # install recommended drivers

sudo apt install nvidia-driver-xxx # or sudo ubuntu-drivers autoinstall

sudo nano /etc/security/limits.conf # insert this text ↓

# *   soft    nproc   65000
# *   hard    nproc   1000000
# *   -    nofile  1048576
# * - memlock unlimited

sudo reboot

wget https://nvidia.github.io/nvidia-docker/gpgkey --no-check-certificate

sudo apt-key add gpgkey

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update && sudo apt-get install -y nvidia-container-toolkit && sudo systemctl restart docker


git clone https://github.com/agladsoft/LocalChatGPT.git

cd LocalChatGPT

sudo docker-compose up
```

## 💻 Get started

To run the program, write

### Docker

```sh
sudo docker-compose up
```

### Venv

```sh
pip install virtualenv

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

python3 scripts/app.py
```

## 🙇‍♂️ Usage

You can change the program settings in the file `__init__.py ` 

To change the scope of the context, change the `CONTEXT_SIZE` variable to `__init__.py`

To upload a new model, add it to the `DICT_REPO_AND_MODELS` variable (the key is the repository, the value is the name of the model in the web interface)

To view the list of selected users, follow the link
http://127.0.0.1:7860

## 👋 Contributing

Please check out the [Contributing to RATH guide](https://docs.kanaries.net/community/contribution-guide)
for guidelines about how to proceed.

Thanks to all contributors :heart:

<a href="https://github.com/agladsoft/LocalChatGPT/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=agladsoft/LocalChatGPT" />
</a>

## ⚖️ License
![alt text](https://seeklogo.com/images/M/MIT-logo-73A348B3DB-seeklogo.com.png)

This project is under the MIT License. See the [LICENSE](https://github.com/gogs/gogs/blob/main/LICENSE) file for the full license text.