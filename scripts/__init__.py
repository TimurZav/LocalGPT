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


os.environ["OPENROUTER_API_KEY"] = ""
FAVICON_PATH: str = 'https://i.ibb.co/3CVGPf7/1681038242chatgpt-logo-png.png'
QUERY_SYSTEM_PROMPT: str = "Вы, помощник по документам - полезный, уважительный и честный ассистент. " \
                     "Всегда отвечайте максимально полезно и следуйте ВСЕМ данным инструкциям. " \
                     "Не спекулируйте и не выдумывайте информацию. " \
                     "Отвечайте на вопросы, ссылаясь на контекст."

LLM_SYSTEM_PROMPT: str = "Вы, помощник по документам — полезный и честный ассистент. " \
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
MODELS: list = ["openai/gpt-4o-mini", "deepseek/deepseek-r1:free", "meta-llama/llama-3.3-70b-instruct:free"]
MODEL_AUDIO = "openai/whisper-large-v3-turbo"
EMBEDDER_NAME: str = "intfloat/multilingual-e5-large"
MAX_NEW_TOKENS: int = 1500

IP_ADDRESS = f"http://{socket.gethostbyname(socket.gethostname())}:8001"

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))

# ChatGPT-style icons (using Unicode emojis as fallback)
CHATGPT_ICONS = {
    'new_chat': '💬',
    'delete': '🗑️', 
    'edit': '✏️',
    'more': '⋯',
    'history': '📚',
    'settings': '⚙️'
}
DATA_DIR: str = "../data"
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
DB_DIR: str = os.path.join(ABS_PATH, f"{DATA_DIR}/chroma")
DATABASE_URL: str = f"sqlite:///{DB_DIR}/users_data.db"
DATABASE_DATA_URL: str = "postgresql://admin:admin@localhost:5432/mydatabase"
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
/* Simple UI with ChatGPT style */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* Sidebar */
.sidebar-container {
    background: var(--background-fill-primary);
    border: 1px solid var(--border-color-primary);
    border-radius: var(--radius-lg);
    padding: 16px;
    box-shadow: var(--shadow-drop);
}

.sidebar-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 16px;
    padding: 8px 12px;
    color: #1a1a1a;
    font-weight: 600;
    font-size: 14px;
}

.new-dialog-btn {
    background: #10a37f !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    width: 100% !important;
    margin-bottom: 12px !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
}

.new-dialog-btn:hover {
    background: #0d8a6b !important;
    transform: translateY(-1px);
}

.dialog-item {
    background: var(--background-fill-primary);
    border: 1px solid var(--border-color-primary);
    border-radius: var(--radius-lg);
    padding: 12px;
    margin: 4px 0;
    cursor: pointer;
    transition: all 0.2s ease;
    animation: slideIn 0.2s ease-out;
}

.dialog-item:hover {
    background: var(--background-fill-secondary);
    border-color: var(--color-accent);
}

.dialog-item.selected {
    background: var(--color-accent) !important;
    color: white !important;
    border-color: var(--color-accent) !important;
}

.delete-btn {
    background: #ef4444 !important;
    color: white !important;
    border: none !important;
}

.delete-btn:hover {
    background: #dc2626 !important;
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateX(-10px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

@media (min-width: 1024px) {
    .modal-container.svelte-7knbu5 {
        max-width: 50% !important;
    }
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

JS_CHATGPT_STYLE: str = """
function() {
    console.log('ChatGPT style loading...');
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
