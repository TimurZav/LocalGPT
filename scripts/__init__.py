import os
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


FAVICON_PATH: str = 'https://i.ibb.co/DGGPZBG/logo.png'
QUERY_SYSTEM_PROMPT: str = "Вы, Макар - полезный, уважительный и честный ассистент. " \
                     "Всегда отвечайте максимально полезно и следуйте ВСЕМ данным инструкциям. " \
                     "Не спекулируйте и не выдумывайте информацию. " \
                     "Отвечайте на вопросы, ссылаясь на контекст."

LLM_SYSTEM_PROMPT: str = "Вы, Макар - полезный, уважительный и честный ассистент."

MODES: list = ["ВНД", "Свободное общение", "Получение документов"]
CONTEXT_SIZE = 4000
SYSTEM_TOKEN: int = 1788
USER_TOKEN: int = 1404
BOT_TOKEN: int = 9225
LINEBREAK_TOKEN: int = 13

ROLE_TOKENS: dict = {
    "user": USER_TOKEN,
    "bot": BOT_TOKEN,
    "system": SYSTEM_TOKEN
}

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

MODEL: str = "saiga2_7b_gguf"
MODEL_Q: str = "model-q2_K.gguf"

REPO: str = f"https://huggingface.co/IlyaGusev/{MODEL}/resolve/main/{MODEL_Q}"
MODEL_NAME = f"{MODEL}/{MODEL_Q}"


EMBEDDER_NAME: str = "intfloat/multilingual-e5-large"

MAX_NEW_TOKENS: int = 1500

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(ABS_PATH, "../chroma")
MODELS_DIR = os.path.join(ABS_PATH, "../models")
LOGGING_DIR: str = os.path.join(ABS_PATH, "../logging")
if not os.path.exists(LOGGING_DIR):
    os.mkdir(LOGGING_DIR)
DATA_QUESTIONS = os.path.join(ABS_PATH, "../data_questions")
if not os.path.exists(DATA_QUESTIONS):
    os.mkdir(DATA_QUESTIONS)

IMAGES = os.path.join(ABS_PATH, "static/img")
if not os.path.exists(IMAGES):
    os.mkdir(IMAGES)
AVATAR_USER = f"{IMAGES}/icons8-человек-96.png"
AVATAR_BOT = f"{IMAGES}/icons8-bot-96.png"
LOGIN_ICON = f"{IMAGES}/login.png"
LOGOUT_ICON = f"{IMAGES}/logout.png"

SOURCES_SEPARATOR = "\n\n Документы: \n"
MESSAGE_LOGIN = "Введите логин и пароль, чтобы войти"

FILES_DIR = os.path.join(ABS_PATH, "../upload_files")
os.makedirs(FILES_DIR, exist_ok=True)
os.chmod(FILES_DIR, 0o0777)
os.environ['GRADIO_TEMP_DIR'] = FILES_DIR

BLOCK_CSS = """

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

"""


JS = """
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


LOCAL_STORAGE = """
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


LOG_FORMAT: str = "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
DATE_FTM: str = "%d/%B/%Y %H:%M:%S"


def get_stream_handler():
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
