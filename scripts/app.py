import re
import uuid
import copy
import ollama
import os.path
import chromadb
import tempfile
import subprocess
import pandas as pd
import gradio as gr
from re import Pattern
from __init__ import *
from datetime import datetime
from gradio_modal import Modal
from tinydb import TinyDB, where
from functions.functions import *
from collections import defaultdict
from langchain.docstore.document import Document
from typing import List, Optional, Tuple, Iterator
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger: logging.getLogger = get_logger(os.path.basename(__file__).replace(".py", "_") + str(datetime.now().date()))


class SystemPromptManager:
    def __init__(self):
        self.mode = MODES[0]
        self.system_prompt = ""

    def set_system_prompt(self, system_prompt_input: str) -> None:
        """
        Setting prompt.
        :param system_prompt_input: Prompt.
        :return:
        """
        self.system_prompt = system_prompt_input

    def set_current_mode(self, mode: str) -> gr.update:
        """
        Setting prompt.
        :param mode: Mode.
        :return:
        """
        self.mode = mode
        self.set_system_prompt(self._get_default_system_prompt(mode))
        # Update placeholder and allow interaction if default system prompt is set
        if self.system_prompt:
            return gr.update(placeholder=self.system_prompt, interactive=True)
        # Update placeholder and disable interaction if no default system prompt is set
        else:
            return gr.update(placeholder=self.system_prompt, interactive=False)

    @staticmethod
    def _get_default_system_prompt(mode: str) -> str:
        """
        Returning prompt of mode.
        :param mode: Mode.
        :return: Prompt.
        """
        return QUERY_SYSTEM_PROMPT if mode == "DB" else LLM_SYSTEM_PROMPT


class AnalyticsManager:
    def __init__(self):
        self.tiny_db: TinyDB = TinyDB(f'{QUESTIONS}/tiny_db.json', indent=4, ensure_ascii=False)

    def get_analytics(self) -> pd.DataFrame:
        """
        Retrieves and returns analytics data from the database as a sorted DataFrame.

        This method fetches all data entries from the `tiny_db` database, converts them into a DataFrame,
        and sorts the records by the 'Старт обработки запроса' (Request Processing Start) column in
        descending order if this column is present. If the column is missing, it returns the DataFrame unsorted.

        :return: A DataFrame containing analytics data, optionally sorted by 'Старт обработки запроса'
        in descending order.
        """
        try:
            return pd.DataFrame(self.tiny_db.all()).sort_values('Старт обработки запроса', ascending=False)
        except KeyError:
            return pd.DataFrame(self.tiny_db.all())

    def update_message_analytics(self, messages: List[dict], analyse=None):
        """
        Updates or inserts analytics data for the latest message in the database.

        This function processes the last message in a list of message-answer pairs (`messages`). If the message
        already exists in the database, it updates the stored answer, increments the repetition count, and
        optionally adds a rating (`analyse`). If the message is new, it inserts a new record with the current
        timestamp. Finally, it returns the updated analytics DataFrame.

        :param messages: List of tuples where each tuple is a (message, answer) pair.
        :param analyse: Optional; rating to assign to the message-answer pair. If not provided, defaults to None.
        :return: A DataFrame containing the latest analytics data.
        """
        message = messages[-2]["content"] if messages else None
        answer = messages[-1]["content"] if message else None
        filter_query = where('Сообщения') == message
        if result := self.tiny_db.search(filter_query):
            if analyse is None:
                self.tiny_db.update({
                    'Ответы': answer,
                    'Количество повторений': result[0]['Количество повторений'] + 1,
                    'Старт обработки запроса': str(datetime.now())
                }, cond=filter_query)
            else:
                self.tiny_db.update({'Оценка ответа': analyse}, cond=filter_query)
                gr.Info("Отзыв ответу поставлен")
        elif message is not None:
            self.tiny_db.insert({
                'Сообщения': message,
                'Ответы': answer,
                'Количество повторений': 1,
                'Оценка ответа': None,
                'Старт обработки запроса': str(datetime.now())
            })
        return self.get_analytics()


class AuthManager:
    def __init__(self, document_manager):
        self.document_manager: DocumentManager = document_manager

    @staticmethod
    def login(username: str, password: str) -> dict:
        """
        Sends a login request to obtain an access token for the provided user credentials.

        This function takes in a username and password, then sends a POST request to the
        authentication endpoint to retrieve an access token. If authentication is successful,
        the access token is returned along with a success flag. In case of failure, an error
        message is logged, and the function returns failure information with an error message.

        :param username: The username of the user attempting to authenticate.
        :param password: The password of the user attempting to authenticate.
        :return: A dictionary containing:
                 - "access_token": The access token if authentication is successful, else None.
                 - "is_success": Boolean indicating success (True) or failure (False).
                 - "message": Error message if authentication fails, else not included.
        """
        try:
            response = requests.post(
                f"http://{IP_ADDRESS}/token",
                data={"username": username, "password": password},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=10  # Adding a timeout
            )
            if response.status_code == 200:
                return {"access_token": response.json().get("access_token"), "is_success": True}
            error_detail = response.json().get("detail", "Unknown error")
            logger.error(f"Login failed with status {response.status_code}: {error_detail}")
            return {"access_token": None, "is_success": False, "message": error_detail}
        except requests.RequestException as e:
            logger.error(f"Error during login request: {e}")
            return {"access_token": None, "is_success": False, "message": "Request failed"}

    def update_user_ui_state(self, local_data: Optional[dict]):
        """
        Retrieves current user information and updates the user interface based on login status.

        Validates the `local_data` to check if the user is authenticated. If authenticated, retrieves
        user details from the server and prepares the UI to reflect the logged-in state. In case of failed
        authentication or a logged-out state, it updates the UI to reflect this accordingly.

        :param local_data: A dictionary with local user data, including an access token and login status.
        :return: A list of UI updates for interface components based on user login status.
        """
        if isinstance(local_data, dict) and local_data.get("is_success", False):
            response = requests.get(
                f"http://{IP_ADDRESS}/users/me",
                headers={"Authorization": f"Bearer {local_data['access_token']}"}
            )
            logger.info(f"User is {response.json().get('username')}")
            is_logged_in = response.status_code == 200
        else:
            is_logged_in = False

        obj_tabs = [local_data] + [gr.update(visible=is_logged_in) for _ in range(3)]
        if is_logged_in:
            obj_tabs.append(gr.update(value="Выйти", icon=LOGOUT_ICON))
        else:
            obj_tabs.append(gr.update(value="Войти", icon=LOGIN_ICON))
        obj_tabs.append(gr.update(visible=not is_logged_in))
        if isinstance(local_data, dict):
            obj_tabs.append(local_data.get("message", MESSAGE_LOGIN))
        else:
            obj_tabs.append(MESSAGE_LOGIN)
        obj_tabs.append(self.document_manager.list_ingested_documents())
        return obj_tabs

    def toggle_login_state(self, local_data: Optional[dict], login_btn: gr.component):
        """
        Handles user login/logout functionality and updates the UI accordingly.

        This function checks the current user's login status using `local_data`. If the user is logged in,
        it updates the UI to reflect a logged-out state, changing the button to "Login." If the user
        is not logged in, it shows the login button and adjusts the visibility of other UI components.

        :param local_data: A dictionary containing local user data, which may include an access token.
        :param login_btn: The Gradio component representing the login button.
        :return: A list of UI updates to be applied based on the user's login state.
        """
        data = self.update_user_ui_state(local_data)
        is_logged_in = isinstance(data[0], dict) and data[0].get("access_token")

        obj_tabs = [gr.update(visible=not is_logged_in)] + [gr.update(visible=False) for _ in range(3)]
        obj_tabs.append(gr.update(value="Войти" if is_logged_in else login_btn))

        return obj_tabs


class DocumentManager:
    def __init__(self):
        self.embeddings: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDER_NAME,
            cache_folder=MODELS_DIR
        )
        self.collection: str = "all-documents"
        self.db: Optional[Chroma] = None

    def initialize_database(self) -> Chroma:
        """
        Loads the database connection using a persistent Chroma client.

        This method initializes a PersistentClient for Chroma, specifying the
        path to the database directory. It then creates and returns a Chroma
        instance that can be used to interact with the specified collection
        using the defined embedding function.

        :return: An instance of Chroma connected to the specified collection.
        """
        client = chromadb.PersistentClient(path=DB_DIR)
        return Chroma(
            client=client,
            collection_name=self.collection,
            embedding_function=self.embeddings,
        )

    @staticmethod
    def load_document_from_file(file_path: str) -> Document:
        """
        Loads a single document from the specified file path.

        This method checks the file extension to ensure it is supported. If the
        extension is valid, it initializes the appropriate loader class and
        loads the document.

        :param file_path: The path to the document file to be loaded.
        :return: An instance of the loaded Document.
        :raises FileNotFoundError: If the specified file cannot be found or loaded.
        :raises ValueError: If the file extension is not supported by the loader.
        """
        ext: str = os.path.splitext(file_path)[1]
        if ext not in LOADER_MAPPING:
            raise ValueError(f"Unsupported file extension: {ext}")

        try:
            loader_class, loader_args = LOADER_MAPPING[ext]
            loader = loader_class(file_path, **loader_args)
            return loader.load()[0]
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise FileNotFoundError(f"Failed to load document at {file_path}") from e

    @staticmethod
    def _process_text(page_content: str) -> str:
        """
        Processes the input text by removing unnecessary lines and formatting it into a more readable form.

        This method filters out lines that are too short or empty, then joins
        the remaining lines into a single string. If the resulting text is shorter
        than 10 characters, an empty string is returned.

        :param page_content: The input string containing the text to be processed.
        :return: A cleaned and formatted version of the input text. Returns an
                 empty string if the processed text is less than 10 characters long.
        """
        lines: list = page_content.split("\n")
        lines = [line for line in lines if len(line.strip()) > 2]
        page_content = "\n".join(lines).strip()
        return "" if len(page_content) < 10 else page_content

    def update_documents(self, fixed_documents: List[Document], ids: List[str]) -> tuple[bool, str]:
        """
        Updates existing documents in the database if their filenames match the uploaded documents.

        This method checks for duplicate filenames between the uploaded documents and
        the existing documents in the database. If duplicates are found, the existing
        documents will be deleted before the new documents are added.

        :param fixed_documents: A list of Document objects to be uploaded.
        :param ids: A list of identifiers corresponding to the documents being uploaded.
        :return: A tuple containing a boolean indicating whether the update was successful,
                 and a message providing feedback on the operation.
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

    def _filter_valid_documents(self, documents: List[Document]) -> List[Document]:
        """
        Filters out documents with insufficient content after processing.
        :param documents: Upload documents.
        :return: Valid documents.
        """
        valid_documents: list = []
        for doc in documents:
            doc.page_content = self._process_text(doc.page_content)
            if doc.page_content:  # Only append if there's valid content
                valid_documents.append(doc)
        return valid_documents

    def index_documents(
        self,
        file_paths: List[tempfile.TemporaryFile],
        chunk_size: int,
        chunk_overlap: int
    ):
        """
        Build an index from the provided document file paths by loading, processing,
        and splitting the documents into manageable chunks.

        :param file_paths: A list of temporary file paths from which to load documents.
        :param chunk_size: The maximum size of each chunk of text after splitting.
        :param chunk_overlap: The number of overlapping characters between chunks to maintain context.
        :return: A warning message indicating the number of fragments loaded and readiness for queries,
                 or any warnings encountered during the update process.
        """
        load_documents: List[Document] = [self.load_document_from_file(path.name) for path in file_paths]
        text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        documents = text_splitter.split_documents(load_documents)
        fixed_documents = self._filter_valid_documents(documents)

        ids: List[str] = [
            f"{os.path.basename(doc.metadata['source']).replace('.txt', '')}{i}"
            for i, doc in enumerate(fixed_documents)
        ]
        is_updated, file_warning = self.update_documents(fixed_documents, ids)
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

    def retrieve_documents(
        self,
        history: List[dict],
        collection_radio: str,
        k_documents: int,
        uid: str
    ) -> Tuple[str, list]:
        """
        Retrieves relevant documents from the database based on the user's most recent message
        and formats them for display, including document URLs and similarity scores.

        This function performs a similarity search on the user's latest message within a specific
        database collection, and returns a formatted string containing the retrieved documents
        along with their similarity scores. If there are no relevant documents or conditions are
        not met, an appropriate message and an empty list are returned.

        :param history: The conversation history as a list of message pairs (user, bot responses).
        :param collection_radio: The selected collection mode for document retrieval.
        :param k_documents: The number of top documents to retrieve based on similarity.
        :param uid: The unique identifier for the current session, used for logging.
        :return: A tuple with a formatted string of retrieved documents and a list of their similarity scores.
        """
        if (
            not self.db
            or collection_radio != MODES[0]
            or not history
            or history[-1]["role"] != "user"
        ):
            return "Появятся после задавания вопросов", []

        last_user_message = history[-1][0]
        docs = self.db.similarity_search_with_score(last_user_message, k_documents)
        scores: list = []
        data = defaultdict(str)

        for doc in docs:
            url = (
                f"""<a href="file/{doc[0].metadata["source"]}" target="_blank" 
                rel="noopener noreferrer">{os.path.basename(doc[0].metadata["source"])}</a>"""
            )
            document: str = f"Document - {url} ↓"
            score: float = round(doc[1], 2)
            scores.append(score)
            data[document] += f"\n\nScore: {score}, Text: {doc[0].page_content}"

        list_data: list = [f"{doc}\n\n{page_content}" for doc, page_content in data.items()]
        logger.info(f"Retrieved context from database for collection '{collection_radio}' [uid - {uid}]")

        if not list_data:
            return "No documents found in the database", scores

        return "\n\n\n".join(list_data), scores

    def list_ingested_documents(self):
        """
        Loads the database and retrieves a list of ingested document filenames.

        This method initializes or reloads the database connection, retrieves document metadata,
        and compiles a list of the filenames of previously ingested documents, returning them
        in a format suitable for UI updates.

        :return: An update object for UI elements with the current list of ingested document filenames.
        """
        self.db = self.initialize_database()
        files = {
            os.path.basename(ingested_document["source"])
            for ingested_document in self.db.get()["metadatas"]
        }
        return gr.update(choices=list(files))

    def delete_documents(self, documents: list):
        """
        Deletes specified documents from the database.

        This method takes a list of document filenames to delete, searches the database for matching
        document entries, and deletes those that match. It then returns an updated list of the
        remaining ingested document filenames.

        :param documents: List of document filenames (without paths) to delete from the database.
        :return: An update object for the UI element containing the list of remaining ingested documents.
        """
        try:
            all_documents: dict = self.db.get()
            if for_delete_ids := [
                doc_id
                for ingested_document, doc_id in zip(all_documents["metadatas"], all_documents["ids"])
                if os.path.basename(ingested_document["source"]) in documents
            ]:
                self.db.delete(for_delete_ids)
            return self.list_ingested_documents()
        except Exception as e:
            logger.error(f"Error during document deletion: {e}")
            return gr.update(choices=[])


class LocalGPT:
    def __init__(self):
        self.load_or_initialize_model()
        self._queue: int = 0
        self.document_manager: DocumentManager = DocumentManager()
        self.analytics_manager: AnalyticsManager = AnalyticsManager()
        self.auth_manager: AuthManager = AuthManager(self.document_manager)
        self.prompt_manager: SystemPromptManager = SystemPromptManager()

    @staticmethod
    def load_or_initialize_model():
        """
        Loads and initializes the Llama model from a specified repository.

        This method creates necessary directories for model storage if they do not
        already exist. It checks if the model file is present; if not, it downloads
        the model from the specified repository. Finally, it initializes and returns
        a Llama model instance with the specified configuration.

        :return: An instance of the Llama model initialized with the specified parameters.
        """
        try:
            subprocess.run(["ollama", "pull", MODEL], check=True)
            logger.info(f"The model {MODEL} has been successfully downloaded")
        except subprocess.CalledProcessError:
            logger.info(f"The model {MODEL} could not be downloaded")

    @staticmethod
    def generate_chat_completion(
        history: List[dict],
        retrieved_docs: str,
        mode: str,
        uid: str
    ):
        """
        Generate a chat completion response based on user history and retrieved documents.

        This function constructs a message prompt for the language model, incorporating user history,
        context from retrieved documents, and various parameters that influence the response generation.

        :param history: A list of tuples containing the conversation history, where each tuple consists of
                        the user's message and the assistant's response.
        :param retrieved_docs: A string containing documents retrieved for context, which may include
                               HTML anchor tags.
        :param mode: A string indicating the mode of response generation.
        :param uid: A unique identifier for the user, used for logging purposes.
        :return: A generator for the chat completion response and a list of files extracted from the
                 retrieved documents.
        """
        last_user_message: str = history[-1].get("content")
        files = re.findall(r'<a\s+[^>]*>(.*?)</a>', retrieved_docs)
        for file in files:
            retrieved_docs = re.sub(fr'<a\s+[^>]*>{file}</a>', file, retrieved_docs)
        if retrieved_docs and mode == MODES[0]:
            last_user_message = (
                f"Контекст: {retrieved_docs}\n\nИспользуя только контекст, ответь на вопрос: "
                f"{last_user_message}"
            )

        pair_count = 0  # Счетчик пар "user-assistant"
        temp_history = []  # Временный список для обратного добавления диалогов
        for message in reversed(history):
            if message["role"] == "user" and pair_count == 0:
                continue
            if message["role"] == "user" and isinstance(message["content"], tuple):
                temp_history.append({
                    "role": "user",
                    "content": temp_history[-1]["content"],
                    "images": [message["content"][0]]
                })
            elif message["role"] == "user":
                temp_history.append({
                    "role": "user",
                    "content": message["content"]
                })
            if message["role"] == "assistant":
                pair_count += 1
                temp_history.append({
                    "role": "assistant",
                    "content": message["content"]
                })
            if pair_count == 3:
                break

        dialog_history: List[dict] = list(reversed(temp_history))
        messages = [
            *dialog_history
        ]
        if len(history) > 1 and isinstance(history[-2].get("content"), tuple):
            messages.append({
                "role": "user",
                "content": last_user_message,
                "images": [history[-2].get("content")[0]]
            })
        else:
            messages.append({
                "role": "user",
                "content": last_user_message
            })

        logger.info(f"The question has been fully formed [uid - {uid}]")
        stream = ollama.chat(
            model=MODEL,
            messages=messages,
            stream=True,
        )
        return stream, files

    @staticmethod
    def _add_source_references(
        history: List[dict],
        scores: List[float],
        files: List[str],
        partial_text: str,
        threshold: float = 0.44
    ) -> List[dict]:
        """
        Appends file source references to the final response text based on score thresholds and
        updates conversation history.
        This method adds a list of file references to the response text if files are provided, adjusting based on score
        threshold conditions. The updated text is appended to the most recent assistant message
        in the conversation history.
        :param history: List representing conversation history as pairs of user and assistant messages.
        :param scores: List of floats representing confidence scores associated with each file, determining if
                       file sources should be appended.
        :param files: List of file names or identifiers to include as sources in the response.
        :param partial_text: The assistant's partial response text to which sources will be appended if files exist.
        :param threshold: The score threshold to determine whether all sources are appended or only the top source.
        :return: Updated conversation history with appended source information if conditions are met.
        """
        if files:
            partial_text += SOURCES_SEPARATOR
            sources_text = [f"{index}. {source}" for index, source in enumerate(files, start=1)]
            if scores and scores[0] < threshold:
                partial_text += "\n\n\n".join(sources_text)
            elif scores:
                partial_text += sources_text[0]
            history[-1]["content"] = partial_text
        return history

    def generate_response_stream(
        self,
        history: List[dict],
        mode: str,
        retrieved_docs: str,
        scores: List[float],
        uid: str
    ) -> Iterator[List[List[str]]]:
        """
        Generates a response based on user input, context, and retrieved documents.

        This function orchestrates the process of generating a response by interacting
        with the chat completion generator.
        The response is built incrementally, updating the conversation history as tokens are received.
        In case of an error (e.g., due to large context size), an error message is appended to the response.
        Once complete, source file references are appended to the final output.

        :param history: List of conversation pairs (user input and bot responses).
        :param mode: Operation mode, which influences the use of context in responses.
        :param retrieved_docs: Relevant documents retrieved to help answer the user query.
        :param scores: List of scores associated with retrieved documents for source filtering.
        :param uid: Unique identifier for the current user session, useful for logging and tracking.
        :return: Yields updated conversation history after each token generated, and the final response with sources.
        """
        logger.info(f"Preparing to generate a response based on context and history [uid - {uid}]")
        if not history or not history[-1].get("role"):
            yield history[:-1]
            return

        partial_text = ""
        logger.info(f"Beginning response generation [uid - {uid}]")
        stream, files = self.generate_chat_completion(
            history=history,
            retrieved_docs=retrieved_docs,
            mode=mode,
            uid=uid
        )
        history.append({"role": "assistant", "content": None})
        buffer = ""
        for chunk in stream:
            buffer += chunk['message']['content']
            history[-1]["content"] = buffer
            yield history

        logger.info(f"Response generation completed [uid - {uid}]")
        yield self._add_source_references(history, scores, files, partial_text)
        self._queue -= 1
        _ = self.analytics_manager.update_message_analytics(history)

    def add_user_message(self, message: dict, history: Optional[List]):
        """
        Adds a new user message to the conversation history and generates a unique session identifier.

        This function appends the user's input message to the conversation history and increments the
        queue counter to indicate a pending response generation. If history is not provided, a new
        conversation history is initialized.

        :param message: The user's input message to be added to the conversation history.
        :param history: The existing conversation history as a list of message pairs (user, bot responses).
                        Each pair is a list, with the second item initially set to None for new user messages.
        :return: A tuple containing an empty string (for response text), the updated conversation history,
                 and a unique session ID (uid).
        """
        uid = uuid.uuid4()
        logger.info(f"Processing the question. Queue - {self._queue}. UID - [{uid}]")
        if history is None:
            history = []
        if message["files"]:
            history.append({"role": "user", "content": message["files"]})
        history.append({"role": "user", "content": message["text"]})
        self._queue += 1
        logger.info(f"The question has been processed. UID - [{uid}]")
        return "", history, uid

    def launch_ui(self):
        """
        Launch the main user interface for the LocalGPT application.

        This method sets up the Gradio interface, defining the layout and components such as tabs, buttons,
        input fields, and various interactive elements. It includes functionality for user authentication,
        document uploading, chat interaction, settings adjustments, and logging.

        The interface consists of the following sections:
        - Chat: For user interaction with the virtual assistant.
        - Documents: For uploading and managing documents.
        - Settings: For configuring various parameters.
        - Logs: For displaying chat analytics.

        Returns:
            gr.Blocks: The Gradio Blocks instance for the LocalGPT application UI.
        """
        with gr.Blocks(
            title="LocalGPT",
            theme=gr.themes.ocean.Ocean(),
            css=BLOCK_CSS
        ) as demo:
            # Ваш логотип и текст заголовка
            logo_svg = f'<img src="{FAVICON_PATH}" width="48px" style="display: inline">'
            header_html = f"""<h1><center>{logo_svg} Виртуальный ассистент</center></h1>"""

            with gr.Row():
                gr.HTML(header_html)
                login_btn = gr.DuplicateButton(
                    "Войти", variant="primary", size="lg", elem_id="login_btn", icon=LOGOUT_ICON
                )

            uid = gr.State(None)
            scores = gr.State(None)
            local_data = gr.JSON({}, visible=False)

            with gr.Tab("Чат"):
                with gr.Row():
                    with gr.Column():
                        collection_radio = gr.Radio(
                            choices=MODES,
                            value=self.prompt_manager.mode,
                            show_label=False
                        )

                with gr.Row():
                    with gr.Column(scale=10):
                        chatbot = gr.Chatbot(
                            label=f"LLM: {MODEL}",
                            height=500,
                            type="messages",
                            show_copy_button=True,
                            show_share_button=True,
                            avatar_images=(
                                AVATAR_USER,
                                AVATAR_BOT
                            )
                        )

                with gr.Row():
                    with gr.Column(scale=20):
                        msg = gr.MultimodalTextbox(
                            label="Отправить сообщение",
                            placeholder="👉 Напишите запрос",
                            show_label=False
                        )

                with gr.Row(elem_id="buttons"):
                    like = gr.Button(value="👍 Понравилось")
                    dislike = gr.Button(value="👎 Не понравилось")
                    stop_btn = gr.Button(value="🛑 Остановить")
                    clear = gr.Button(value="🗑️ Очистить")

                with gr.Row():
                    gr.Markdown(
                        "<center>Ассистент может допускать ошибки, поэтому рекомендуем проверять важную информацию. "
                        "Ответы также не являются призывом к действию</center>"
                    )

            with gr.Tab("Документы", visible=False) as documents_tab:
                with gr.Row():
                    with gr.Column(scale=3):
                        upload_files = gr.Files(
                            label="Загрузка документов",
                            file_count="multiple"
                        )
                        file_warning = gr.Markdown("Фрагменты ещё не загружены!")

                    with gr.Column(scale=7):
                        files_selected = gr.Dropdown(
                            choices=None,
                            label="Выберите файлы для удаления",
                            value="",
                            multiselect=True
                        )
                        delete = gr.Button("🧹 Удалить", variant="primary")

            with gr.Tab("Настройки", visible=False) as settings_tab:
                with gr.Row(elem_id="model_selector_row"):
                    models = [MODEL]
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

                with gr.Accordion("Системный промпт", open=False):
                    system_prompt = gr.Textbox(
                        placeholder=QUERY_SYSTEM_PROMPT,
                        lines=5,
                        show_label=False
                    )
                    # On blur, set system prompt to use in queries
                    system_prompt.blur(
                        self.prompt_manager.set_system_prompt,
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
                            value=self.analytics_manager.get_analytics,  # type: ignore
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
                fn=self.auth_manager.login,
                inputs=[login, password],
                outputs=[local_data]
            ).success(
                fn=self.auth_manager.update_user_ui_state,
                inputs=[local_data],
                outputs=[local_data, documents_tab, settings_tab, logging_tab, login_btn, modal, message_login]
            ).success(
                fn=None,
                inputs=[local_data],
                outputs=None,
                js="(v) => {setStorage('access_token', v)}"
            )

            login_btn.click(
                fn=self.auth_manager.toggle_login_state,
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
                fn=self.prompt_manager.set_current_mode,
                inputs=collection_radio,
                outputs=system_prompt
            )

            # Upload files
            upload_files.upload(
                fn=self.document_manager.index_documents,
                inputs=[upload_files, chunk_size, chunk_overlap],
                outputs=[file_warning],
                queue=True
            ).success(
                fn=self.document_manager.list_ingested_documents,
                outputs=files_selected
            )

            # Delete documents from db
            delete.click(
                fn=self.document_manager.delete_documents,
                inputs=files_selected,
                outputs=[files_selected]
            )

            # Pressing Enter
            click_msg_event = msg.submit(
                fn=self.add_user_message,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot, uid],
                queue=False,
            ).success(
                fn=self.document_manager.retrieve_documents,
                inputs=[chatbot, collection_radio, k_documents, uid],
                outputs=[retrieved_docs, scores],
                queue=True,
            ).success(
                fn=self.generate_response_stream,
                inputs=[chatbot, collection_radio, retrieved_docs, scores, uid],
                outputs=chatbot,
                queue=True
            )

            # Like
            like.click(
                fn=self.analytics_manager.update_message_analytics,
                inputs=[chatbot, like],
                outputs=[analytics],
                queue=True,
            )

            # Dislike
            dislike.click(
                fn=self.analytics_manager.update_message_analytics,
                inputs=[chatbot, dislike],
                outputs=[analytics],
                queue=True,
            )

            # Clear history
            clear.click(
                fn=lambda: None,
                inputs=None,
                outputs=chatbot,
                queue=False,
                js=JS
            )

            # Stop generation
            stop_btn.click(
                fn=None,
                inputs=None,
                outputs=None,
                cancels=[click_msg_event]
            )

            demo.load(
                fn=self.auth_manager.update_user_ui_state,
                inputs=[local_data],
                outputs=[
                    local_data,
                    documents_tab,
                    settings_tab,
                    logging_tab,
                    login_btn,
                    modal,
                    message_login,
                    files_selected
                ],
                js=LOCAL_STORAGE
            )

        demo.queue(max_size=128, api_open=False)
        return demo
