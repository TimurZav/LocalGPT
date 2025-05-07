import os
import socket
import logging
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)


FAVICON_PATH: str = 'https://i.ibb.co/3CVGPf7/1681038242chatgpt-logo-png.png'
QUERY_SYSTEM_PROMPT: str = "Вы, Макар - полезный, уважительный и честный ассистент. " \
                     "Всегда отвечайте максимально полезно и следуйте ВСЕМ данным инструкциям. " \
                     "Не спекулируйте и не выдумывайте информацию. " \
                     "Отвечайте на вопросы, ссылаясь на контекст."

LLM_SYSTEM_PROMPT: str = "Вы, Макар, — полезный и честный ассистент. " \
                         "Данные от функций надежны, но могут быть нерелевантны. Анализируйте их в контексте вопроса " \
                         "и дополняйте своим ответом, чтобы он был полным и полезным."

MODES: list = ["RAG", "Поиск", "Свободное общение"]
CONTEXT_SIZE = 4000

LOADER_MAPPING: dict = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

IP_MODEL: str = "http://localhost:11434"
LOGIN_SERVER: str = "Test"
PASSWORD_SERVER: str = "Test"
MODELS: list = ["granite3.2", "llama3.2-vision"]
MODEL_AUDIO = "openai/whisper-large-v3-turbo"
EMBEDDER_NAME: str = "intfloat/multilingual-e5-large"
MAX_NEW_TOKENS: int = 1500

IP_ADDRESS = f"http://{socket.gethostbyname(socket.gethostname())}:8001"

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = "../data"
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
DB_DIR: str = os.path.join(ABS_PATH, f"{DATA_DIR}/chroma")
DATABASE_URL: str = f"sqlite:///{DB_DIR}/users_data.db"
DATABASE_DATA_URL: str = "mysql+pymysql://myuser:mypassword@localhost:3306/mydatabase"
MODELS_DIR: str = os.path.join(ABS_PATH, f"{DATA_DIR}/models")
LOGGING_DIR: str = os.path.join(ABS_PATH, f"{DATA_DIR}/logging")
if not os.path.exists(LOGGING_DIR):
    os.mkdir(LOGGING_DIR)
QUESTIONS: str = os.path.join(ABS_PATH, f"{DATA_DIR}/questions")
if not os.path.exists(QUESTIONS):
    os.mkdir(QUESTIONS)

IMAGES: str = os.path.join(ABS_PATH, "static/img")
if not os.path.exists(IMAGES):
    os.mkdir(IMAGES)
AVATAR_USER: str = f"{IMAGES}/icons8-человек-96.png"
AVATAR_BOT: str = f"{IMAGES}/icons8-bot-96.png"
LOGIN_ICON: str = f"{IMAGES}/login.png"
LOGOUT_ICON: str = f"{IMAGES}/logout.png"

SOURCES_SEPARATOR = "\n\n Документы: \n"
MESSAGE_LOGIN = "Введите логин и пароль, чтобы войти"

FILES_DIR: str = os.path.join(ABS_PATH, f"{DATA_DIR}/upload_files")
os.makedirs(FILES_DIR, exist_ok=True)
os.chmod(FILES_DIR, 0o0777)
os.environ['GRADIO_TEMP_DIR'] = FILES_DIR

BLOCK_CSS: str = """

#buttons button {
    min-width: min(120px,100%);
}

/* Применяем стили для td */
tr focus {
    user-select: all; /* Разрешаем выделение текста */
}

/* Применяем стили для ячейки span внутри td */
tr span {
    user-select: all; /* Разрешаем выделение текста */
}

.message-bubble-border.svelte-12dsd9j.svelte-12dsd9j.svelte-12dsd9j {
  border-style: none;
}

.user {
    background: #2042b9;
    color: white;
}

@media (min-width: 1024px) {
    .modal-container.svelte-7knbu5 {
        max-width: 50% !important
    }
}

.gap.svelte-1m1obck {
    padding: 4%
}

#login_btn {
    width: 250px;
    height: 40px;
}

.icon-button-wrapper {
    display: none;
}

.hide {
    display: none;
}

.upload-button {
    pointer-events: none;
    cursor: not-allowed;
    opacity: 0.5;
}

.enable {
    pointer-events: auto;
    cursor: pointer;
    opacity: 1;
}

"""


JS: str = """
function disable_btn() {
    var elements = document.getElementsByClassName('wrap default minimal svelte-1occ011 translucent');

    for (var i = 0; i < elements.length; i++) {
        if (elements[i].classList.contains('generating') || !elements[i].classList.contains('hide')) {
            // Выполнить любое действие здесь
            console.log('Элемент содержит класс generating');
            // Например:
            document.getElementById('component-35').disabled = true
            setTimeout(() => { document.getElementById('component-35').disabled = false }, 180000);
        }
    }
}
"""


LOCAL_STORAGE: str = """
function() {
    globalThis.setStorage = (key, value) => {
        localStorage.setItem(key, JSON.stringify(value))
    }
    globalThis.removeStorage = (key) => {
        localStorage.removeItem(key)
    }
    globalThis.getStorage = (key, value) => {
        return JSON.parse(localStorage.getItem(key))
    }
    const access_token = getStorage('access_token')
    return [access_token];
}
"""

JS_MODEL_TOGGLE: str = """
function toggleUploadButton(model) {
    const uploadButton = document.querySelector('.upload-button');
    if (model.includes("llama3.2-vision")) {
        uploadButton.classList.add('enable');
    } else {
        uploadButton.classList.remove('enable');
    }
    return [model]
}
"""


LOG_FORMAT: str = "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
DATE_FTM: str = "%d/%B/%Y %H:%M:%S"


def get_stream_handler() -> logging.StreamHandler:
    stream_handler: logging.StreamHandler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    return stream_handler


def get_logger(name: str) -> logging.getLogger:
    logger: logging.getLogger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(get_stream_handler())
    logger.setLevel(logging.INFO)
    return logger
