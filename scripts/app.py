import re
import uuid
import os.path
import chromadb
import requests
import tempfile
import threading
import pandas as pd
import gradio as gr
from re import Pattern
from __init__ import *
from llama_cpp import Llama
from gradio_modal import Modal
from template import create_doc
from tinydb import TinyDB, where
from logging_custom import FileLogger
from typing import List, Optional, Tuple
from datetime import datetime, timedelta, date
from langchain.docstore.document import Document
from huggingface_hub.file_download import http_get
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


logger = get_logger(os.path.basename(__file__).replace(".py", "_") + str(datetime.now().date()))
f_logger = FileLogger(__name__, f"{LOGGING_DIR}/answers_bot.log", mode='a', level=logging.INFO)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LocalChatGPT:

    def __init__(self):
        self.llama_model: Optional[Llama] = self.initialize_app()
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDER_NAME, cache_folder=MODELS_DIR)
        self.db: Optional[Chroma] = None
        self.collection: str = "all-documents"
        self.load_db()
        self.mode: str = MODES[0]
        self.system_prompt = self._get_default_system_prompt(self.mode)
        self.semaphore = threading.Semaphore()
        self._queue = 0
        self.tiny_db = TinyDB(f'{DATA_QUESTIONS}/tiny_db.json', indent=4, ensure_ascii=False)

    @staticmethod
    def initialize_app() -> Llama:
        """
        Загружаем все модели из списка.
        :return:
        """
        os.makedirs(MODELS_DIR, exist_ok=True)
        final_model_path = os.path.join(MODELS_DIR, MODEL_NAME)
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)

        if not os.path.exists(final_model_path):
            with open(final_model_path, "wb") as f:
                http_get(REPO, f)

        return Llama(
            n_gpu_layers=43,
            model_path=final_model_path,
            n_ctx=CONTEXT_SIZE,
            n_parts=1,
        )

    @staticmethod
    def load_single_document(file_path: str) -> Document:
        """
        Загружаем один документ.
        :param file_path:
        :return:
        """
        ext: str = "." + file_path.rsplit(".", 1)[-1]
        assert ext in LOADER_MAPPING
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()[0]

    @staticmethod
    def upload_files(files: List[tempfile.TemporaryFile]) -> List[str]:
        """

        :param files:
        :return:
        """
        return [f.name for f in files]

    @staticmethod
    def process_text(text: str) -> Optional[str]:
        """

        :param text:
        :return:
        """
        lines: list = text.split("\n")
        lines = [line for line in lines if len(line.strip()) > 2]
        text = "\n".join(lines).strip()
        return "" if len(text) < 10 else text

    def update_text_db(
        self,
        fixed_documents: List[Document],
        ids: List[str]
    ) -> tuple[bool, str]:
        """

        :param fixed_documents:
        :param ids:
        :return:
        """
        data: dict = self.db.get()
        files_db = {os.path.basename(dict_data['source']) for dict_data in data["metadatas"]}
        files_load = {os.path.basename(dict_data.metadata["source"]) for dict_data in fixed_documents}
        if same_files := files_load & files_db:
            gr.Warning("Файлы " + ", ".join(same_files) + " повторяются, поэтому они будут обновлены")
            for file in same_files:
                pattern: Pattern[str] = re.compile(fr'{file.replace(".txt", "")}\d*$')
                self.db.delete([x for x in data['ids'] if pattern.match(x)])
            self.db = self.db.from_documents(
                documents=fixed_documents,
                embedding=self.embeddings,
                ids=ids,
                persist_directory=DB_DIR,
                collection_name=self.collection,
            )
            file_warning = f"Загружено {len(fixed_documents)} фрагментов! Можно задавать вопросы."
            return True, file_warning
        return False, "Фрагменты ещё не загружены!"

    def build_index(
        self,
        file_paths: List[str],
        chunk_size: int,
        chunk_overlap: int
    ):
        """

        :param file_paths:
        :param chunk_size:
        :param chunk_overlap:
        :return:
        """
        load_documents: List[Document] = [self.load_single_document(path) for path in file_paths]
        text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        documents = text_splitter.split_documents(load_documents)
        fixed_documents: List[Document] = []
        for doc in documents:
            doc.page_content = self.process_text(doc.page_content)
            if not doc.page_content:
                continue
            fixed_documents.append(doc)

        ids: List[str] = [
            f"{os.path.basename(doc.metadata['source']).replace('.txt', '')}{i}"
            for i, doc in enumerate(fixed_documents)
        ]
        is_updated, file_warning = self.update_text_db(fixed_documents, ids)
        if is_updated:
            return file_warning
        self.db = self.db.from_documents(
            documents=fixed_documents,
            embedding=self.embeddings,
            ids=ids,
            persist_directory=DB_DIR,
            collection_name=self.collection,
        )
        file_warning = f"Загружено {len(fixed_documents)} фрагментов! Можно задавать вопросы."
        os.chmod(FILES_DIR, 0o0777)
        return file_warning

    @staticmethod
    def _get_default_system_prompt(mode: str) -> str:
        return QUERY_SYSTEM_PROMPT if mode == "DB" else LLM_SYSTEM_PROMPT

    def _set_system_prompt(self, system_prompt_input: str) -> None:
        self.system_prompt = system_prompt_input

    def _set_current_mode(self, mode: str):
        self.mode = mode
        self._set_system_prompt(self._get_default_system_prompt(mode))
        # Update placeholder and allow interaction if default system prompt is set
        if self.system_prompt:
            return gr.update(placeholder=self.system_prompt, interactive=True)
        # Update placeholder and disable interaction if no default system prompt is set
        else:
            return gr.update(placeholder=self.system_prompt, interactive=False)

    @staticmethod
    def calculate_end_date(history):
        long_days = re.findall(r"\d{1,4} д", history[-1][0])
        list_dates = []
        for day in long_days:
            day = int(day.replace(" д", ""))
            start_dates = re.findall(r"\d{1,2}[.]\d{1,2}[.]\d{2,4}", history[-1][1])
            for date_ in start_dates:
                list_dates.append(date_)
                end_date = datetime.strptime(date_, '%d.%m.%Y') + timedelta(days=day)
                end_date = end_date.strftime('%d.%m.%Y')
                list_dates.append(end_date)
                return [[f"Начало отпуска - {list_dates[0]}. Конец отпуска - {list_dates[1]}", None]]

    def get_dates_in_question(self, history, generator, mode):
        if mode == MODES[2]:
            partial_text = ""
            for token in generator:
                for data in token["choices"]:
                    letters = data["delta"].get("content", "")
                    partial_text += letters
                    f_logger.finfo(letters)
                    history[-1][1] = partial_text
            return self.calculate_end_date(history)

    def get_message_generator(self, history, retrieved_docs, mode, top_k, top_p, temp, uid):
        model = self.llama_model
        last_user_message = history[-1][0]
        pattern = r'<a\s+[^>]*>(.*?)</a>'
        files = re.findall(pattern, retrieved_docs)
        for file in files:
            retrieved_docs = re.sub(fr'<a\s+[^>]*>{file}</a>', file, retrieved_docs)
        if retrieved_docs and mode == MODES[1]:
            last_user_message = f"Контекст: {retrieved_docs}\n\nИспользуя только контекст, ответь на вопрос: " \
                                f"{last_user_message}"
        elif mode == MODES[2]:
            last_user_message = f"{last_user_message}\n\n" \
                                f"Сегодня {datetime.now().strftime('%d.%m.%Y')} число. " \
                                f"Если в контексте не указан год, то пиши {date.today().year}. " \
                                f"Напиши ответ только так, без каких либо дополнений: " \
                                f"Прошу предоставить ежегодный оплачиваемый отпуск с " \
                                f"(дата начала отпуска в формате DD.MM.YYYY) по " \
                                f"(дата окончания отпуска в формате DD.MM.YYYY)."
        logger.info(f"Вопрос был полностью сформирован [uid - {uid}]")
        f_logger.finfo(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - Вопрос: {history[-1][0]} - "
                       f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
        history_user = [
            {"role": "user", "content": user_message}
            for user_message, _ in history[-4:-1]
        ]
        generator = model.create_chat_completion(
            messages=[
                {
                    "role": "system", "content": self.system_prompt
                },
                *history_user,
                {
                    "role": "user", "content": last_user_message
                },

            ],
            stream=True,
            temperature=temp,
            top_k=top_k,
            top_p=top_p
        )
        return model, generator, files

    @staticmethod
    def get_list_files(history, mode, scores, files, partial_text):
        if files:
            partial_text += SOURCES_SEPARATOR
            sources_text = [
                f"{index}. {source}"
                for index, source in enumerate(files, start=1)
            ]
            threshold = 0.44
            if scores and scores[0] < threshold:
                partial_text += "\n\n\n".join(sources_text)
            elif scores:
                partial_text += sources_text[0]
            history[-1][1] = partial_text
        elif mode == MODES[2]:
            file = create_doc(partial_text, "Титова", "Сергея Сергеевича", "Руководитель отдела",
                              "Отдел организационного развития")
            partial_text += f'\n\n\nФайл: {file}'
            history[-1][1] = partial_text
        return history

    def bot(self, history, mode, retrieved_docs, top_p, top_k, temp, scores, uid):
        """

        :param history:
        :param mode:
        :param retrieved_docs:
        :param top_p:
        :param top_k:
        :param temp:
        :param scores:
        :param uid:
        :return:
        """
        logger.info(f"Подготовка к генерации ответа. Формирование полного вопроса на основе контекста и истории "
                    f"[uid - {uid}]")
        self.semaphore.acquire()
        if not history or not history[-1][0]:
            yield history[:-1]
            self.semaphore.release()
            return
        model, generator, files = self.get_message_generator(history, retrieved_docs, mode, top_k, top_p, temp, uid)
        partial_text = ""
        logger.info(f"Начинается генерация ответа [uid - {uid}]")
        f_logger.finfo(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - Ответ: ")
        if message := self.get_dates_in_question(history, generator, mode):
            model, generator, files = self.get_message_generator(message, retrieved_docs, mode, top_k, top_p, temp, uid)
        elif mode == MODES[2]:
            model, generator, files = self.get_message_generator(history, retrieved_docs, mode, top_k, top_p, temp, uid)
        try:
            token: dict
            for token in generator:
                for data in token["choices"]:
                    letters = data["delta"].get("content", "")
                    partial_text += letters
                    f_logger.finfo(letters)
                    history[-1][1] = partial_text
                    yield history
        except Exception as ex:
            logger.error(f"Error - {ex}")
            partial_text += "\nСлишком большой контекст. " \
                            "Попробуйте уменьшить его или измените количество выдаваемого контекста в настройках"
            history[-1][1] = partial_text
            yield history
        f_logger.finfo(f" - [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n\n")
        logger.info(f"Генерация ответа закончена [uid - {uid}]")
        yield self.get_list_files(history, mode, scores, files, partial_text)
        self._queue -= 1
        self.semaphore.release()

    def user(self, message, history):
        uid = uuid.uuid4()
        logger.info(f"Обработка вопроса. Очередь - {self._queue}. UID - [{uid}]")
        self.semaphore.acquire()
        if history is None:
            history = []
        new_history = history + [[message, None]]
        self._queue += 1
        self.semaphore.release()
        logger.info(f"Закончена обработка вопроса. UID - [{uid}]")
        return "", new_history, uid

    def retrieve(self, history, collection_radio, k_documents: int, uid: str) -> Tuple[str, list]:
        """

        :param history:
        :param collection_radio:
        :param k_documents:
        :param uid:
        :return:
        """
        if not self.db or collection_radio != MODES[0] or not history or not history[-1][0]:
            return "Появятся после задавания вопросов", []
        last_user_message = history[-1][0]
        docs = self.db.similarity_search_with_score(last_user_message, k_documents)
        scores: list = []
        data: dict = {}
        for doc in docs:
            url = f"""<a href="file/{doc[0].metadata["source"]}" target="_blank" 
                rel="noopener noreferrer">{os.path.basename(doc[0].metadata["source"])}</a>"""
            document: str = f'Документ - {url} ↓'
            score: float = round(doc[1], 2)
            scores.append(score)
            if document in data:
                data[document] += "\n\n" + f"Score: {score}, Text: {doc[0].page_content}"
            else:
                data[document] = f"Score: {score}, Text: {doc[0].page_content}"
        list_data: list = [f"{doc}\n\n{text}" for doc, text in data.items()]
        logger.info(f"Получили контекст из базы [uid - {uid}]")
        if not list_data:
            return "Документов в базе нету", scores
        return "\n\n\n".join(list_data), scores

    def ingest_files(self):
        self.load_db()
        files = {
            os.path.basename(ingested_document["source"])
            for ingested_document in self.db.get()["metadatas"]
        }
        return list(files)

    def delete_doc(self, documents: str):
        self.load_db()
        all_documents: dict = self.db.get()
        for_delete_ids: list = []
        list_documents: List[str] = documents.strip().split("\n")
        for ingested_document, doc_id in zip(all_documents["metadatas"], all_documents["ids"]):
            print(ingested_document)
            if os.path.basename(ingested_document["source"]) in list_documents:
                for_delete_ids.append(doc_id)
        if for_delete_ids:
            self.db.delete(for_delete_ids)
        return gr.update(choices=self.ingest_files())

    def get_analytics(self) -> pd.DataFrame:
        try:
            return pd.DataFrame(self.tiny_db.all()).sort_values('Старт обработки запроса', ascending=False)
        except KeyError:
            return pd.DataFrame(self.tiny_db.all())

    def calculate_analytics(self, messages, analyse=None):
        message = messages[-1][0] if messages else None
        answer = messages[-1][1] if message else None
        filter_query = where('Сообщения') == message
        if result := self.tiny_db.search(filter_query):
            if analyse is None:
                self.tiny_db.update(
                    {
                        'Ответы': answer,
                        'Количество повторений': result[0]['Количество повторений'] + 1,
                        'Старт обработки запроса': str(datetime.now())
                    },
                    cond=filter_query
                )
            else:
                self.tiny_db.update({'Оценка ответа': analyse}, cond=filter_query)
                gr.Info("Отзыв ответу поставлен")
        elif message is not None:
            self.tiny_db.insert(
                {'Сообщения': message, 'Ответы': answer, 'Количество повторений': 1, 'Оценка ответа': None,
                 'Старт обработки запроса': str(datetime.now())}
            )
        return self.get_analytics()

    @staticmethod
    def login(username, password):
        """

        :param username:
        :param password:
        :return:
        """
        response = requests.post(
            "http://0.0.0.0:8001/token",
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if response.status_code == 200:
            return {"access_token": response.json()["access_token"], "is_success": True}
        logger.error(response.json()["detail"])
        return {"access_token": None, "is_success": False, "message": response.json()["detail"]}

    @staticmethod
    def get_current_user_info(local_data):
        """

        :param local_data:
        :return:
        """
        if isinstance(local_data, dict) and local_data.get("is_success", False):
            response = requests.get(
                "http://0.0.0.0:8001/users/me",
                headers={"Authorization": f"Bearer {local_data['access_token']}"}
            )
            logger.info(f"User is {response.json().get('username')}")
            is_logged_in = response.status_code == 200
        else:
            is_logged_in = False

        obj_tabs = [local_data] + [gr.update(visible=is_logged_in) for _ in range(3)]
        if is_logged_in:
            obj_tabs.append(gr.update(value="Выйти", icon=str(LOGOUT_ICON)))
        else:
            obj_tabs.append(gr.update(value="Войти", icon=str(LOGIN_ICON)))
        obj_tabs.append(gr.update(visible=not is_logged_in))
        if isinstance(local_data, dict):
            obj_tabs.append(local_data.get("message", MESSAGE_LOGIN))

        return obj_tabs

    def login_or_logout(self, local_data, login_btn):
        """

        :param local_data:
        :param login_btn:
        :return:
        """
        data = self.get_current_user_info(local_data)
        if isinstance(data[0], dict) and data[0].get("access_token"):
            obj_tabs = [gr.update(visible=False)] + [gr.update(visible=False) for _ in range(3)]
            obj_tabs.append(gr.update(value="Войти", icon=str(LOGIN_ICON)))
            return obj_tabs
        obj_tabs = [gr.update(visible=True)] + [gr.update(visible=False) for _ in range(3)]
        obj_tabs.append(login_btn)
        return obj_tabs

    def load_db(self):
        """

        :return:
        """
        client = chromadb.PersistentClient(path=DB_DIR)
        self.db = Chroma(
            client=client,
            collection_name=self.collection,
            embedding_function=self.embeddings,
        )

    def run(self):
        """

        :return:
        """
        with gr.Blocks(
                title="MakarGPT",
                theme=gr.themes.Soft().set(
                    body_background_fill="white",
                    block_background_fill="#e1e5e8",
                    block_label_background_fill="#2042b9",
                    block_label_background_fill_dark="#2042b9",
                    block_label_text_color="white",
                    checkbox_label_background_fill_selected="#1f419b",
                    checkbox_label_background_fill_selected_dark="#1f419b",
                    checkbox_background_color_selected="#111d3d",
                    checkbox_background_color_selected_dark="#111d3d",
                    input_background_fill="#e1e5e8",
                    button_primary_background_fill="#1f419b",
                    button_primary_background_fill_dark="#1f419b",
                    shadow_drop_lg="5px 5px 5px 5px rgb(0 0 0 / 0.1)"
                ),
                css=BLOCK_CSS
        ) as demo:
            # Ваш логотип и текст заголовка
            logo_svg = f'<img src="{FAVICON_PATH}" width="48px" style="display: inline">'
            header_html = f"""<h1><center>{logo_svg} Виртуальный ассистент Рускон (бета-версия)</center></h1>"""

            with gr.Row():
                gr.HTML(header_html)
                login_btn = gr.DuplicateButton("Войти", variant="primary", size="lg", elem_id="login_btn",
                                               icon=str(LOGIN_ICON))

            uid = gr.State(None)
            scores = gr.State(None)
            local_data = gr.JSON({}, visible=False)
            file_paths = gr.State(None)

            with gr.Tab("Чат"):
                with gr.Row():
                    collection_radio = gr.Radio(
                        choices=MODES,
                        value=self.mode,
                        show_label=False
                    )

                with gr.Row():
                    with gr.Column(scale=10):
                        chatbot = gr.Chatbot(
                            label="Диалог",
                            height=500,
                            show_copy_button=True,
                            show_share_button=True,
                            avatar_images=(
                                AVATAR_USER,
                                AVATAR_BOT
                            )
                        )

                with gr.Row():
                    with gr.Column(scale=20):
                        msg = gr.Textbox(
                            label="Отправить сообщение",
                            show_label=False,
                            placeholder="👉 Напишите запрос",
                            container=False
                        )
                    with gr.Column(scale=3, min_width=100):
                        submit = gr.Button("📤 Отправить", variant="primary")

                with gr.Row(elem_id="buttons"):
                    like = gr.Button(value="👍 Понравилось")
                    dislike = gr.Button(value="👎 Не понравилось")
                    clear = gr.Button(value="🗑️ Очистить")

                with gr.Row():
                    gr.Markdown(
                        "<center>Ассистент может допускать ошибки, поэтому рекомендуем проверять важную информацию. "
                        "Ответы также не являются призывом к действию</center>"
                    )

            with gr.Tab("Документы", visible=False) as documents_tab:
                with gr.Row():
                    with gr.Column(scale=3):
                        upload_button = gr.Files(
                            label="Загрузка документов",
                            file_count="multiple"
                        )
                        file_warning = gr.Markdown("Фрагменты ещё не загружены!")

                    with gr.Column(scale=7):
                        list_files = self.ingest_files()
                        files_selected = gr.Dropdown(
                            choices=list_files,
                            label="Выберите файлы для удаления",
                            value=None,
                            multiselect=True
                        )
                        delete = gr.Button("🧹 Удалить", variant="primary")

            with gr.Tab("Настройки", visible=False) as settings_tab:
                with gr.Row(elem_id="model_selector_row"):
                    models = [MODEL_NAME]
                    gr.Dropdown(
                        choices=models,
                        value=models[0],
                        interactive=True,
                        show_label=False,
                        container=False,
                    )
                with gr.Accordion("Параметры", open=False):
                    with gr.Tab(label="Параметры извлечения фрагментов из текста"):
                        k_documents = gr.Slider(
                            minimum=1,
                            maximum=12,
                            value=6,
                            step=1,
                            interactive=True,
                            label="Кол-во фрагментов для контекста"
                        )
                    with gr.Tab(label="Параметры нарезки"):
                        chunk_size = gr.Slider(
                            minimum=128,
                            maximum=1792,
                            value=1408,
                            step=128,
                            interactive=True,
                            label="Размер фрагментов",
                        )
                        chunk_overlap = gr.Slider(
                            minimum=0,
                            maximum=400,
                            value=400,
                            step=10,
                            interactive=True,
                            label="Пересечение"
                        )
                    with gr.Tab(label="Параметры генерации"):
                        top_p = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            interactive=True,
                            label="Top-p",
                        )
                        top_k = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=80,
                            step=5,
                            interactive=True,
                            label="Top-k",
                        )
                        temp = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.1,
                            step=0.1,
                            interactive=True,
                            label="Temp"
                        )

                with gr.Accordion("Системный промпт", open=False):
                    system_prompt = gr.Textbox(
                        placeholder=QUERY_SYSTEM_PROMPT,
                        lines=5,
                        show_label=False
                    )
                    # On blur, set system prompt to use in queries
                    system_prompt.blur(
                        self._set_system_prompt,
                        inputs=system_prompt,
                    )

                with gr.Accordion("Контекст", open=True):
                    with gr.Column(variant="compact"):
                        retrieved_docs = gr.Markdown(
                            value="Появятся после задавания вопросов",
                            label="Извлеченные фрагменты",
                            show_label=True
                        )

            with gr.Tab("Логи диалогов", visible=False) as logging_tab:
                with gr.Row():
                    with gr.Column():
                        analytics = gr.DataFrame(
                            value=self.get_analytics,
                            interactive=False,
                            wrap=True
                        )

            with Modal(visible=False) as modal:
                with gr.Column(variant="panel"):
                    gr.HTML("<h1><center>Вход</center></h1>")
                    message_login = gr.HTML(MESSAGE_LOGIN)
                    login = gr.Textbox(
                        label="Логин",
                        placeholder="Введите логин",
                        show_label=True,
                        max_lines=1
                    )
                    password = gr.Textbox(
                        label="Пароль",
                        placeholder="Введите пароль",
                        show_label=True,
                        type="password"
                    )
                    submit_login = gr.Button("👤 Войти", variant="primary")
                    cancel_login = gr.Button("⛔ Отмена", variant="secondary")

            submit_login.click(
                fn=self.login,
                inputs=[login, password],
                outputs=[local_data]
            ).success(
                fn=self.get_current_user_info,
                inputs=[local_data],
                outputs=[local_data, documents_tab, settings_tab, logging_tab, login_btn, modal, message_login]
            ).success(
                fn=None,
                inputs=[local_data],
                outputs=None,
                js="(v) => {setStorage('access_token', v)}"
            )

            login_btn.click(
                fn=self.login_or_logout,
                inputs=[local_data, login_btn],
                outputs=[modal, documents_tab, settings_tab, logging_tab, login_btn]
            ).success(
                fn=None,
                inputs=None,
                outputs=[local_data],
                js="() => {removeStorage('access_token')}"
            )
            cancel_login.click(
                fn=lambda: Modal(visible=False),
                inputs=None,
                outputs=modal
            )

            collection_radio.change(
                fn=self._set_current_mode, inputs=collection_radio, outputs=system_prompt
            )

            # Upload files
            upload_button.upload(
                fn=self.upload_files,
                inputs=[upload_button],
                outputs=[file_paths],
                queue=True,
            ).success(
                fn=self.build_index,
                inputs=[file_paths, chunk_size, chunk_overlap],
                outputs=[file_warning],
                queue=True
            ).success(
                self.ingest_files,
                outputs=files_selected
            )

            # Delete documents from db
            delete.click(
                fn=self.delete_doc,
                inputs=files_selected,
                outputs=[files_selected]
            )

            # Pressing Enter
            msg.submit(
                fn=self.user,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot, uid],
                queue=False,
            ).success(
                fn=self.retrieve,
                inputs=[chatbot, collection_radio, k_documents, uid],
                outputs=[retrieved_docs, scores],
                queue=True,
            ).success(
                fn=self.bot,
                inputs=[chatbot, collection_radio, retrieved_docs, top_p, top_k, temp, scores, uid],
                outputs=chatbot,
                queue=True
            ).success(
                fn=self.calculate_analytics,
                inputs=chatbot,
                outputs=analytics,
                queue=True,
            )

            # Pressing the button
            submit.click(
                fn=self.user,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot, uid],
                queue=False,
            ).success(
                fn=self.retrieve,
                inputs=[chatbot, collection_radio, k_documents, uid],
                outputs=[retrieved_docs, scores],
                queue=True,
            ).success(
                fn=self.bot,
                inputs=[chatbot, collection_radio, retrieved_docs, top_p, top_k, temp, scores, uid],
                outputs=chatbot,
                queue=True
            ).success(
                fn=self.calculate_analytics,
                inputs=chatbot,
                outputs=analytics,
                queue=True,
            )

            # Like
            like.click(
                fn=self.calculate_analytics,
                inputs=[chatbot, like],
                outputs=[analytics],
                queue=True,
            )

            # Dislike
            dislike.click(
                fn=self.calculate_analytics,
                inputs=[chatbot, dislike],
                outputs=[analytics],
                queue=True,
            )

            # Clear history
            clear.click(lambda: None, None, chatbot, queue=False, js=JS)

            demo.load(
                fn=self.get_current_user_info,
                inputs=[local_data],
                outputs=[local_data, documents_tab, settings_tab, logging_tab, login_btn],
                js=LOCAL_STORAGE
            )

        demo.queue(max_size=128, api_open=False)
        return demo
