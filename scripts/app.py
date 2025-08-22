import re
import uuid
import os.path
import chromadb
import tempfile
import numpy as np
import pandas as pd
import gradio as gr
import soundfile as sf
import glob
from re import Pattern
from __init__ import *
from gradio_modal import Modal
from tinydb import TinyDB, where
from yake import KeywordExtractor
from functions.functions import *
from transformers import pipeline
from neo4j import GraphDatabase
from collections import defaultdict
from tinydb.queries import QueryLike
from claude_code_llm import ClaudeCodeLLM
from datetime import datetime, timedelta
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_neo4j import Neo4jVector, Neo4jGraph
from langchain_community.utilities import SQLDatabase
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.agent_toolkits import create_sql_agent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Optional, Tuple, AsyncGenerator, cast, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from natasha import MorphVocab, Doc, Segmenter, NewsMorphTagger, NewsEmbedding


os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger: logging.getLogger = get_logger(
    str(os.path.basename(__file__).replace(".py", "_") + str(datetime.now().date()))
)


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
        return QUERY_SYSTEM_PROMPT


class AnalyticsManager:
    def __init__(self):
        self.tiny_db: TinyDB = TinyDB(f'{QUESTIONS}/tiny_db.json', indent=4, ensure_ascii=False)
        # Отдельная база для истории диалогов
        self.dialogs_db: TinyDB = TinyDB(f'{QUESTIONS}/dialogs.json', indent=4, ensure_ascii=False)

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

    def save_dialog_session(self, session_id: str, messages: List[dict], title: str = None) -> None:
        """
        Сохранить диалоговую сессию с уникальным ID.
        """
        if not messages:
            return
            
        # Генерируем заголовок на основе первого сообщения пользователя
        if not title:
            first_user_msg = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
            title = (first_user_msg[:50] + "...") if len(first_user_msg) > 50 else first_user_msg
            
        dialog_data = {
            'session_id': session_id,
            'title': title,
            'messages': messages,
            'created_at': str(datetime.now()),
            'updated_at': str(datetime.now())
        }
        
        # Проверяем, существует ли уже диалог с таким ID
        existing = self.dialogs_db.search(where('session_id') == session_id)
        if existing:
            self.dialogs_db.update(dialog_data, where('session_id') == session_id)
        else:
            self.dialogs_db.insert(dialog_data)
    
    def get_all_dialogs(self) -> List[dict]:
        """
        Получить список всех сохраненных диалогов.
        """
        dialogs = self.dialogs_db.all()
        # Сортируем по дате обновления (самые новые первые)
        return sorted(dialogs, key=lambda x: x.get('updated_at', ''), reverse=True)
    
    def load_dialog_session(self, session_id: str) -> Optional[List[dict]]:
        """
        Загрузить конкретную диалоговую сессию.
        """
        result = self.dialogs_db.search(where('session_id') == session_id)
        return result[0]['messages'] if result else None
    
    def delete_dialog_session(self, session_id: str) -> bool:
        """
        Удалить диалоговую сессию.
        """
        removed = self.dialogs_db.remove(where('session_id') == session_id)
        return len(removed) > 0
        
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
        filter_query = cast(QueryLike, where('Сообщения') == message)
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


class VMManager:
    def __init__(self):
        self.server_id: str = "43ba92d7-d3bd-4100-9487-46a3f3ef1db0"
        self.url: str = f"https://api.immers.cloud:8774/v2.1/servers/{self.server_id}"
        self.headers: dict = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent":
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
        }

    def authenticate(self) -> str:
        """
        Authenticates with the cloud API to obtain an authorization token.
        :return: Success message with the token if the request is successful.
            Error message with details if the request fails.
        """
        url: str = "https://api.immers.cloud:5000/v3/auth/tokens"
        payload: dict = {
            "auth": {
                "identity": {
                    "methods": ["password"],
                    "password": {
                        "user": {
                            "name": LOGIN_SERVER,
                            "password": PASSWORD_SERVER,
                            "domain": {
                                "id": "default"
                            }
                        }
                    }
                },
                "scope": {
                    "project": {
                        "name": LOGIN_SERVER,
                        "domain": {
                            "id": "default"
                        }
                    }
                }
            }
        }
        response = requests.post(url, headers=self.headers, json=payload)
        if response.status_code != 201:
            return f"Ошибка: {response.status_code}. Детали ответа: {response.text}"
        os.environ["OS_TOKEN"] = response.headers.get("X-Subject-Token")
        return f"Авторизация успешна! Токен: {os.environ['OS_TOKEN']}"

    def send_action(self, action: str) -> str:
        """
        Sends an action command to the server as a POST request.
        :param action: The action to perform on the server (e.g., "os-start", "os-stop").
        :return: Success message if the request is successful.
            Error message with details if the request fails.
        """
        payload = {action: None}
        response = requests.post(f"{self.url}/action", headers=self.headers, json=payload)

        if response.status_code in {200, 202}:
            return f"Запрос '{action}' выполнен успешно!"
        else:
            return f"Ошибка: {response.status_code}\nДетали: {response.text}"

    def status(self) -> str:
        """
        Retrieves the current status of the server.
        :return: The server's status and last updated date if the request is successful.
            Error message with details if the request fails.
        """
        response = requests.get(self.url, headers=self.headers)
        if response.status_code not in {200, 202}:
            return f"Ошибка: {response.status_code}\nДетали: {response.text}"
        json_data = response.json()['server']
        return f"Статус: '{json_data['status']}'. Последняя дата обновления: " \
               f"{datetime.strptime(json_data['updated'], '%Y-%m-%dT%H:%M:%SZ') + timedelta(hours=3)}"

    def control_vm(self, action: str):
        """
        Controls the server by invoking the corresponding action or method.
        :param action: The action to perform.
        :return: The result of the performed action or an error message for invalid actions.
        """
        actions_map: dict = {
            "Вкл": "os-start",
            "Выкл": "os-stop",
            "Перезагрузить": "reboot",
            "Архивировать": "shelve",
            "Разархивировать": "unshelve",
            "Статус": self.status,
            "Авторизация": self.authenticate
        }

        if action not in actions_map:
            return "Неизвестное действие"

        if action == "Авторизация":
            return actions_map[action]()

        self.headers["X-Auth-Token"] = os.environ['OS_TOKEN']
        if callable(actions_map[action]):
            return actions_map[action]()

        return self.send_action(actions_map[action])


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
                f"{IP_ADDRESS}/token",
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

    def update_user_ui_state(self, local_data: Optional[dict], is_visible: bool = True):
        """
        Retrieves current user information and updates the user interface based on login status.

        Validates the `local_data` to check if the user is authenticated. If authenticated, retrieves
        user details from the server and prepares the UI to reflect the logged-in state. In case of failed
        authentication or a logged-out state, it updates the UI to reflect this accordingly.

        :param local_data: A dictionary with local user data, including an access token and login status.
        :param is_visible:
        :return: A list of UI updates for interface components based on user login status.
        """
        if isinstance(local_data, dict) and local_data.get("is_success", False):
            response = requests.get(
                f"{IP_ADDRESS}/users/me",
                headers={"Authorization": f"Bearer {local_data['access_token']}"}
            )
            logger.info(f"User is {response.json().get('username')}")
            is_logged_in = response.status_code == 200
            is_visible = False
        else:
            is_logged_in = False

        obj_tabs: List[Union[gr.update, None, str]] = [local_data] + [gr.update(visible=is_logged_in) for _ in range(3)]
        if is_logged_in:
            obj_tabs.append(gr.update(value="Выйти", icon=LOGOUT_ICON))
        else:
            obj_tabs.append(gr.update(value="Войти", icon=LOGIN_ICON))
        obj_tabs.append(gr.update(visible=is_visible))
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
        obj_tabs.append(gr.update(value="Войти", icon=LOGIN_ICON if is_logged_in else login_btn))

        return obj_tabs


class DocumentManager:
    def __init__(self):
        self.embeddings: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDER_NAME,
            cache_folder=MODELS_DIR
        )
        self.collection: str = "all-documents"
        
        # Neo4j connection parameters
        self.neo4j_url = "bolt://localhost:7687"
        self.neo4j_username = "neo4j"
        self.neo4j_password = "localgpt123"
        
        # Initialize Neo4j connections
        self.neo4j_vector: Optional[Neo4jVector] = None
        self.neo4j_graph: Optional[Neo4jGraph] = None
        self.graph_driver = None
        
        # Legacy Chroma support (for backward compatibility)
        self.db: Optional[Chroma] = None
        
        # Other components
        self.segmenter: Segmenter = Segmenter()
        self.morph_vocab: MorphVocab = MorphVocab()
        self.morph_tagger: NewsMorphTagger = NewsMorphTagger(NewsEmbedding())
        self.cache: dict = {}
        self.log_file_path: str = ""  # Path to the log file
        self.log_entries: List[str] = []  # Cached log entries
        self.csv_logs_data: pd.DataFrame = pd.DataFrame()  # CSV logs data
        self.data_path: str = "/home/timur/PycharmWork/LocalGPT/data2"  # Path to data folder
        
        # Initialize Neo4j components
        self._initialize_neo4j()

    def load_log_file(self, file_path: str) -> None:
        """
        Loads log entries from a .txt file.
        
        :param file_path: Path to the log file
        """
        try:
            self.log_file_path = file_path
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by lines and filter out empty lines
            self.log_entries = [line.strip() for line in content.split('\n') if line.strip()]
            logger.info(f"Loaded {len(self.log_entries)} log entries from {file_path}")
        except Exception as e:
            logger.error(f"Error loading log file {file_path}: {e}")
            self.log_entries = []

    def _initialize_neo4j(self):
        """
        Initialize Neo4j connections and components for GraphRAG.
        """
        try:
            # Initialize Graph Database driver
            self.graph_driver = GraphDatabase.driver(
                self.neo4j_url,
                auth=(self.neo4j_username, self.neo4j_password)
            )
            
            # Initialize Neo4j Graph for structured queries
            self.neo4j_graph = Neo4jGraph(
                url=self.neo4j_url,
                username=self.neo4j_username,
                password=self.neo4j_password
            )
            
            # Initialize Neo4j Vector Store for embeddings with safer approach
            self.neo4j_vector = Neo4jVector.from_existing_graph(
                embedding=self.embeddings,
                url=self.neo4j_url,
                username=self.neo4j_username,
                password=self.neo4j_password,
                index_name="document_embeddings",
                node_label="Document",
                text_node_properties=["content", "title"],
                embedding_node_property="embedding"
            )
            
            if self.neo4j_vector:
                logger.info("Neo4j GraphRAG initialized successfully")
            else:
                logger.warning("Neo4j Vector Store not available, will use graph-only approach")
            
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j: {e}")
            # Fallback to Chroma if Neo4j fails
            self._initialize_chroma_fallback()
    
    def _initialize_chroma_fallback(self):
        """
        Fallback to Chroma if Neo4j initialization fails.
        """
        try:
            client = chromadb.PersistentClient(path=DB_DIR)
            self.db = Chroma(
                client=client,
                collection_name=self.collection,
                embedding_function=self.embeddings,
            )
            logger.info("Fallback to Chroma database initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Chroma fallback: {e}")
    
    def initialize_database(self):
        """
        Initialize the database (Neo4j or Chroma fallback).
        """
        if self.neo4j_vector is None:
            self._initialize_neo4j()
        
        if self.neo4j_vector is None and self.db is None:
            self._initialize_chroma_fallback()
        
        return self.neo4j_vector or self.db

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

    def _create_graph_relationships(self, documents: List[Document]):
        """
        Create relationships between documents and entities in Neo4j graph.
        """
        if not self.graph_driver:
            return
        
        with self.graph_driver.session() as session:
            for i, doc in enumerate(documents):
                # Extract entities and relationships using keyword extraction
                try:
                    kw_extractor = KeywordExtractor(lan="ru", n=3, dedupLim=0.3, top=10)
                    keywords = kw_extractor.extract_keywords(doc.page_content)
                    
                    # Create document node
                    session.run(
                        """
                        MERGE (d:Document {id: $doc_id})
                        SET d.content = $content,
                            d.source = $source,
                            d.title = $title,
                            d.chunk_index = $chunk_index
                        """,
                        doc_id=f"doc_{i}_{doc.metadata.get('source', 'unknown')}",
                        content=doc.page_content,
                        source=doc.metadata.get('source', 'unknown'),
                        title=os.path.basename(doc.metadata.get('source', 'unknown')),
                        chunk_index=i
                    )
                    
                    # Create entity nodes and relationships
                    for keyword, score in keywords[:5]:  # Top 5 keywords
                        session.run(
                            """
                            MERGE (e:Entity {name: $entity})
                            SET e.type = 'keyword'
                            WITH e
                            MATCH (d:Document {id: $doc_id})
                            MERGE (d)-[r:CONTAINS]->(e)
                            SET r.score = $score
                            """,
                            entity=keyword,
                            doc_id=f"doc_{i}_{doc.metadata.get('source', 'unknown')}",
                            score=float(score)
                        )
                except Exception as e:
                    logger.warning(f"Failed to create relationships for document {i}: {e}")
    
    def update_documents(self, fixed_documents: List[Document], ids: List[str]) -> tuple[bool, str]:
        """
        Updates existing documents in the database (Neo4j or Chroma fallback).
        """
        try:
            if self.neo4j_vector:
                # Neo4j approach
                # Check for existing documents and remove duplicates
                existing_docs = self._get_existing_document_names()
                new_files = {os.path.basename(doc.metadata["source"]) for doc in fixed_documents}
                
                if same_files := new_files & existing_docs:
                    gr.Warning("Файлы " + ", ".join(same_files) + " повторяются, поэтому они будут обновлены")
                    self._delete_documents_by_names(list(same_files))
                
                # Add documents to Neo4j vector store
                self.neo4j_vector.add_documents(fixed_documents, ids=ids)
                
                # Create graph relationships
                self._create_graph_relationships(fixed_documents)
                
                file_warning = f"Загружено {len(fixed_documents)} фрагментов в Neo4j GraphRAG! Можно задавать вопросы."
                return True, file_warning
            
            elif self.db:
                # Fallback to Chroma
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
                file_warning = f"Загружено {len(fixed_documents)} фрагментов в Chroma! Можно задавать вопросы."
                return True, file_warning
            
            return False, "База данных не инициализирована!"
            
        except Exception as e:
            logger.error(f"Error updating documents: {e}")
            return False, f"Ошибка при обновлении документов: {str(e)}"

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
        Build an index from the provided document file paths using Neo4j GraphRAG or Chroma fallback.

        :param file_paths: A list of temporary file paths from which to load documents.
        :param chunk_size: The maximum size of each chunk of text after splitting.
        :param chunk_overlap: The number of overlapping characters between chunks to maintain context.
        :return: A warning message indicating the number of fragments loaded and readiness for queries.
        """
        try:
            # Load and process documents
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
            
            # Try to update using Neo4j or Chroma
            is_updated, file_warning = self.update_documents(fixed_documents, ids)
            if is_updated:
                return file_warning
            
            # Fallback: create new collection
            if self.neo4j_vector:
                # Neo4j approach
                self.neo4j_vector.add_documents(fixed_documents, ids=ids)
                self._create_graph_relationships(fixed_documents)
                file_warning = f"Загружено {len(fixed_documents)} фрагментов в Neo4j GraphRAG! Можно задавать вопросы."
            elif self.db:
                # Chroma approach
                self.db = self.db.from_documents(
                    documents=fixed_documents,
                    embedding=self.embeddings,
                    ids=ids,
                    persist_directory=DB_DIR,
                    collection_name=self.collection,
                )
                file_warning = f"Загружено {len(fixed_documents)} фрагментов в Chroma! Можно задавать вопросы."
            else:
                return "Ошибка: не удалось инициализировать базу данных!"
            
            try:
                os.chmod(FILES_DIR, 0o0777)
            except:
                pass  # Ignore permission errors
                
            return file_warning
            
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            return f"Ошибка при индексации документов: {str(e)}"

    def lemmatize(self, texts: List[str]) -> List[str]:
        """
        Лемматизация текста с использованием кэша.
        :param texts: Исходный текст.
        :return: Лемматизированный текст.
        """
        result = []
        for text_ in texts:
            doc = Doc(text_)
            doc.segment(self.segmenter)
            doc.tag_morph(self.morph_tagger)

            lemmatized_tokens = []
            for token in doc.tokens:
                if token.text in self.cache:
                    lemma = self.cache[token.text]
                else:
                    token.lemmatize(self.morph_vocab)
                    lemma = token.lemma
                    self.cache[token.text] = lemma
                lemmatized_tokens.append(lemma)

            result.append(" ".join(lemmatized_tokens))
        return result

    def search_docs(self, sentence: str) -> List[str]:
        """
        Legacy method - returns all logs for backward compatibility.
        Use get_all_logs() for clearer intent.

        :param sentence: Предложение или ключевое слово (не используется).
        :return: Все загруженные логи.
        """
        return self.log_entries if self.log_entries else []
    
    def _get_existing_document_names(self) -> set:
        """
        Get existing document names from Neo4j.
        """
        if not self.graph_driver:
            return set()
        
        try:
            with self.graph_driver.session() as session:
                result = session.run("MATCH (d:Document) RETURN DISTINCT d.title as title")
                return {record["title"] for record in result if record["title"]}
        except Exception as e:
            logger.error(f"Error getting existing documents: {e}")
            return set()
    
    def _delete_documents_by_names(self, filenames: List[str]):
        """
        Delete documents by filenames from Neo4j.
        """
        if not self.graph_driver:
            return
        
        try:
            with self.graph_driver.session() as session:
                for filename in filenames:
                    session.run(
                        "MATCH (d:Document {title: $title}) DETACH DELETE d",
                        title=filename
                    )
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
    
    def get_rag_context(self, query: str, k_documents: int = 6) -> Tuple[str, List[str]]:
        """
        Get RAG context from documents using Neo4j GraphRAG or Chroma fallback.
        
        :param query: User query for RAG search
        :param k_documents: Number of documents to retrieve
        :return: Tuple of (clean_context, sources)
        """
        try:
            if self.neo4j_vector:
                # Neo4j GraphRAG approach
                docs = self.neo4j_vector.similarity_search_with_score(query, k=k_documents)
                if not docs:
                    return "", []
                
                # Get additional context from graph relationships
                graph_context = self._get_graph_context(query)
                
                clean_chunks = []
                sources = []
                
                for doc, score in docs:
                    clean_chunks.append(f"Score: {round(score, 2)}\nText: {doc.page_content}")
                    source = doc.metadata.get("source", "unknown")
                    sources.append(os.path.basename(source) if source != "unknown" else "neo4j_graph")
                
                # Add graph context if available
                if graph_context:
                    clean_chunks.append(f"Graph Context:\n{graph_context}")
                    sources.append("neo4j_relationships")
                
                clean_context = "\n\n".join(clean_chunks)
                return clean_context, sources
                
            elif self.neo4j_graph:
                # Neo4j graph-only approach (without vector search)
                graph_context = self._get_graph_context_only(query, k_documents)
                if graph_context:
                    return graph_context, ["neo4j_graph_search"]
                return "", []
                
            elif self.db:
                # Fallback to Chroma
                docs = self.db.similarity_search_with_score(query, k_documents)
                if not docs:
                    return "", []
                    
                clean_chunks = []
                sources = []
                
                for doc in docs:
                    clean_chunks.append(f"Score: {round(doc[1], 2)}\nText: {doc[0].page_content}")
                    sources.append(os.path.basename(doc[0].metadata["source"]))
                
                clean_context = "\n\n".join(clean_chunks)
                return clean_context, sources
            
            return "", []
            
        except Exception as e:
            logger.error(f"Error getting RAG context: {e}")
            return "", []
    
    def _get_graph_context(self, query: str) -> str:
        """
        Get additional context from Neo4j graph relationships.
        """
        if not self.neo4j_graph:
            return ""
        
        try:
            # Extract entities from query
            kw_extractor = KeywordExtractor(lan="ru", n=2, dedupLim=0.5, top=3)
            keywords = kw_extractor.extract_keywords(query)
            
            if not keywords:
                return ""
            
            # Find related entities and documents
            query_cypher = """
            MATCH (d:Document)-[r:CONTAINS]->(e:Entity)
            WHERE e.name IN $keywords
            RETURN d.title as document, e.name as entity, r.score as relevance
            ORDER BY r.score DESC
            LIMIT 5
            """
            
            result = self.neo4j_graph.query(
                query_cypher, 
                {"keywords": [kw[0] for kw in keywords[:3]]}
            )
            
            if result:
                context_parts = []
                for record in result:
                    context_parts.append(
                        f"Document: {record['document']} contains '{record['entity']}' (relevance: {record['relevance']:.2f})"
                    )
                return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting graph context: {e}")
        
        return ""
    
    def _get_graph_context_only(self, query: str, k_documents: int = 6) -> str:
        """
        Get context using only Neo4j graph search (without vector embeddings).
        """
        if not self.neo4j_graph:
            return ""
        
        try:
            # Extract entities from query
            kw_extractor = KeywordExtractor(lan="ru", n=2, dedupLim=0.5, top=5)
            keywords = kw_extractor.extract_keywords(query)
            
            if not keywords:
                # Fallback to full-text search in graph
                query_cypher = """
                MATCH (d:Document)
                WHERE d.content CONTAINS $query_text
                RETURN d.content as content, d.title as source
                ORDER BY size(d.content) DESC
                LIMIT $limit
                """
                
                result = self.neo4j_graph.query(
                    query_cypher,
                    {"query_text": query[:100], "limit": k_documents}
                )
            else:
                # Search by extracted keywords
                query_cypher = """
                MATCH (d:Document)-[r:CONTAINS]->(e:Entity)
                WHERE e.name IN $keywords
                WITH d, AVG(r.score) as avg_score
                ORDER BY avg_score DESC
                LIMIT $limit
                RETURN d.content as content, d.title as source, avg_score
                """
                
                result = self.neo4j_graph.query(
                    query_cypher, 
                    {"keywords": [kw[0] for kw in keywords[:5]], "limit": k_documents}
                )
            
            if result:
                context_parts = []
                for i, record in enumerate(result):
                    score = record.get('avg_score', 0.5)
                    content = record['content'][:500] + "..." if len(record['content']) > 500 else record['content']
                    context_parts.append(
                        f"Score: {score:.2f}\nSource: {record['source']}\nText: {content}"
                    )
                return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting graph-only context: {e}")
        
        return ""

    def retrieve_documents(
        self,
        history: List[dict],
        collection_radio: str,
        k_documents: int,
        uid: str
    ) -> Tuple[str, list]:
        """
        Retrieves relevant documents using GraphRAG search for UI display.
        Uses Neo4j GraphRAG or Chroma fallback.

        :param history: The conversation history as a list of message pairs (user, bot responses).
        :param collection_radio: The selected collection mode for document retrieval.
        :param k_documents: The number of top documents to retrieve based on similarity.
        :param uid: The unique identifier for the current session, used for logging.
        :return: A tuple with formatted RAG documents and similarity scores (for UI display only).
        """
        if (
            collection_radio not in MODES
            or not history
            or history[-1]["role"] != "user"
        ):
            return "Появятся после задавания вопросов", []

        last_user_message = history[-1].get("content")
        
        try:
            # Try Neo4j GraphRAG first
            if self.neo4j_vector:
                docs = self.neo4j_vector.similarity_search_with_score(last_user_message, k=k_documents)
                if docs:
                    scores: list = []
                    data = defaultdict(str)
                    graph_context = self._get_graph_context(last_user_message)

                    for doc, score in docs:
                        source = doc.metadata.get("source", "neo4j_graph")
                        if source != "neo4j_graph":
                            url = f'<a href="file/{source}" target="_blank" rel="noopener noreferrer">{os.path.basename(source)}</a>'
                        else:
                            url = "Neo4j Graph"
                        
                        document: str = f"Document - {url} ↓"
                        score_rounded: float = round(score, 2)
                        scores.append(score_rounded)
                        data[document] += f"\n\nScore: {score_rounded}, Text: {doc.page_content}"
                    
                    # Add graph relationships if available
                    if graph_context:
                        data["Graph Relations ↓"] += f"\n\n{graph_context}"

                    list_data: list = [f"{doc}\n\n{page_content}" for doc, page_content in data.items()]
                    logger.info(f"Retrieved {len(docs)} GraphRAG documents for UI display [uid - {uid}]")
                    
                    return "\n\n\n".join(list_data), scores
                else:
                    return "No relevant documents found in Neo4j GraphRAG", []
            
            # Fallback to Chroma
            elif self.db:
                docs = self.db.similarity_search_with_score(last_user_message, k_documents)
                if docs:
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
                    logger.info(f"Retrieved {len(docs)} Chroma documents for UI display [uid - {uid}]")
                    
                    return "\n\n\n".join(list_data), scores
                else:
                    return "No relevant documents found in Chroma", []
            
            return "База данных не инициализирована", []
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return f"Ошибка при поиске документов: {str(e)}", []
    
    def get_all_logs(self) -> List[str]:
        """
        Returns all log entries without any filtering.
        
        :return: List of all log entries
        """
        return self.log_entries if self.log_entries else []

    def list_ingested_documents(self):
        """
        Retrieves a list of ingested document filenames from Neo4j or Chroma.

        :return: An update object for UI elements with the current list of ingested document filenames.
        """
        try:
            files = set()
            
            if self.neo4j_vector and self.graph_driver:
                # Get files from Neo4j
                with self.graph_driver.session() as session:
                    result = session.run("MATCH (d:Document) RETURN DISTINCT d.title as title")
                    files = {record["title"] for record in result if record["title"]}
            
            elif self.db:
                # Fallback to Chroma
                self.db = self.initialize_database()
                if self.db:
                    files = {
                        os.path.basename(ingested_document["source"])
                        for ingested_document in self.db.get()["metadatas"]
                    }
            
            return gr.update(choices=list(files))
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return gr.update(choices=[])

    def delete_documents(self, documents: list):
        """
        Deletes specified documents from Neo4j or Chroma database.

        :param documents: List of document filenames (without paths) to delete from the database.
        :return: An update object for the UI element containing the list of remaining ingested documents.
        """
        try:
            if self.neo4j_vector and self.graph_driver:
                # Delete from Neo4j
                self._delete_documents_by_names(documents)
                # Also delete from vector store if possible
                # Note: Neo4jVector doesn't have a direct delete by filename method
                # This would require custom implementation
                
            elif self.db:
                # Delete from Chroma
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
    
    def load_log_file_ui(self, file):
        """
        UI handler for loading log files.
        
        :param file: Gradio file object
        :return: Status message
        """
        if file is None:
            return "Файл не выбран"
        
        try:
            self.load_log_file(file.name)
            return f"✅ Загружено {len(self.log_entries)} записей из {os.path.basename(file.name)}"
        except Exception as e:
            logger.error(f"Error loading log file: {e}")
            return f"❌ Ошибка загрузки файла: {str(e)}"
    
    def load_csv_logs_from_data(self, request_time: str, pid: str) -> str:
        """
        Load and filter CSV logs from data folder based on time range and PID.
        
        :param request_time: User request time in ISO format or empty string
        :param pid: User PID filter or empty string
        :return: Status message
        """
        try:
            if not os.path.exists(self.data_path):
                return f"❌ Папка {self.data_path} не найдена"
            
            # Find all CSV files in data folder
            csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
            if not csv_files:
                return "❌ CSV файлы не найдены в папке data"
            
            # Load and combine all CSV files
            all_dfs = []
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    if 'pid' in df.columns and 'Time' in df.columns:
                        df['source_file'] = os.path.basename(csv_file)
                        all_dfs.append(df)
                except Exception as e:
                    logger.warning(f"Не удалось загрузить {csv_file}: {e}")
                    continue
            
            if not all_dfs:
                return "❌ Не найдено CSV файлов с корректной структурой (pid, Time)"
            
            # Combine all data
            combined_df = pd.concat(all_dfs, ignore_index=True, sort=False)
            logger.info(f"Загружено {len(combined_df)} записей из {len(all_dfs)} CSV файлов")
            
            # Convert Time column to datetime
            combined_df['Time'] = pd.to_datetime(combined_df['Time'], errors='coerce')
            
            # Filter by PID if specified
            if pid and pid.strip():
                pid = pid.strip()
                combined_df = combined_df[combined_df['pid'] == pid]
                if combined_df.empty:
                    return f"❌ Не найдено записей для PID: {pid}"
            
            # Filter by time range if request_time is specified
            if request_time:
                try:
                    # Parse request time (Unix timestamp from Gradio)
                    request_dt = pd.to_datetime(request_time, unit='s', utc=True).tz_convert('Europe/Moscow')
                    # Calculate 2 days back from request time
                    two_days_back = request_dt - timedelta(days=2)
                    
                    # Remove timezone info for comparison
                    request_dt = request_dt.tz_localize(None)
                    two_days_back = two_days_back.tz_localize(None)
                    
                    # Filter logs in time range (2 days back to request time)
                    combined_df = combined_df[
                        (combined_df['Time'] >= two_days_back) & 
                        (combined_df['Time'] <= request_dt)
                    ]
                    
                    if combined_df.empty:
                        return f"❌ Не найдено записей в диапазоне от {two_days_back} до {request_dt}"
                        
                except Exception as e:
                    return f"❌ Ошибка парсинга времени запроса: {e}"
            
            # Sort by time
            combined_df = combined_df.sort_values('Time')
            
            # Store the filtered data
            self.csv_logs_data = combined_df
            
            # Convert to log entries format for compatibility with existing system
            log_entries = []
            for _, row in combined_df.iterrows():
                # Create a log entry string with key information
                log_parts = []
                log_parts.append(f"PID: {row['pid']}")
                log_parts.append(f"Event: {row.get('EventName', 'N/A')}")
                log_parts.append(f"Time: {row['Time']}")
                log_parts.append(f"Source: {row['source_file']}")
                
                # Add other relevant columns
                for col in row.index:
                    if col not in ['pid', 'EventName', 'Time', 'source_file'] and pd.notna(row[col]):
                        log_parts.append(f"{col}: {row[col]}")
                
                log_entries.append(" | ".join(log_parts))
            
            self.log_entries = log_entries
            
            # Prepare summary
            summary_parts = [
                f"✅ Загружено {len(combined_df)} записей CSV логов",
                f"📁 Файлов: {len(all_dfs)}",
            ]
            
            if pid:
                summary_parts.append(f"👤 PID: {pid}")
            
            if request_dt:
                summary_parts.append(f"⏰ Период: 2 дня назад от {request_dt}")
                
            unique_pids = combined_df['pid'].nunique() if 'pid' in combined_df.columns else 0
            summary_parts.append(f"👥 Уникальных PID: {unique_pids}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error loading CSV logs: {e}")
            return f"❌ Ошибка загрузки CSV логов: {str(e)}"
    
    def get_csv_logs_summary(self) -> dict:
        """
        Get summary information about loaded CSV logs.
        
        :return: Dictionary with summary information
        """
        if self.csv_logs_data.empty:
            return {"status": "no_data", "message": "CSV логи не загружены"}
        
        try:
            df = self.csv_logs_data
            summary = {
                "status": "loaded",
                "total_records": len(df),
                "unique_pids": df['pid'].nunique() if 'pid' in df.columns else 0,
                "unique_events": df['EventName'].nunique() if 'EventName' in df.columns else 0,
                "time_range": {
                    "start": df['Time'].min().strftime("%Y-%m-%d %H:%M:%S") if 'Time' in df.columns and not df['Time'].isna().all() else "N/A",
                    "end": df['Time'].max().strftime("%Y-%m-%d %H:%M:%S") if 'Time' in df.columns and not df['Time'].isna().all() else "N/A"
                },
                "source_files": df['source_file'].unique().tolist() if 'source_file' in df.columns else []
            }
            return summary
        except Exception as e:
            return {"status": "error", "message": f"Ошибка анализа данных: {str(e)}"}


class AudioManager:
    def __init__(self):
        pass
        # self.pipeline = pipeline("automatic-speech-recognition", model=MODEL_AUDIO)

    @staticmethod
    def add_to_stream(audio: list, in_stream: list) -> tuple:
        """
        Adds a new audio segment to the current stream.
        :param audio: A new audio segment (frequency and data).
        :param in_stream: The current data flow.
        :return: Updated data stream.
        """
        if in_stream is None:
            ret = audio
        else:
            ret = (audio[0], np.concatenate((in_stream[1], audio[1])))
        return audio, ret

    @staticmethod
    def stop_recording() -> gr.component:
        """
        Stops recording and resets the current stream.
        :return: The Gradio component.
        """
        return gr.Audio(value=None, streaming=True)

    def transcribe(self, messages: dict, inputs: list) -> tuple:
        """
        Converts audio files to text.
        :param messages:
        :param inputs: Input audio file (frequency and data array).
        :return: Decrypted text and None (data reset).
        """
        if inputs is None:
            raise gr.Error(
                "No audio file submitted! Please upload or record an audio file before submitting your request"
            )
        sr, y = inputs
        # Convert to mono if stereo
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))
        messages["text"] = self.pipeline({"sampling_rate": sr, "raw": y})["text"]

        return messages, None

    def transcribe_from_file(self, file_path: str) -> str:
        """
        Downloads an audio file and transcribes it into text.
        
        ::param file_path: Path to the audio file
        :return: Transcribed text or error message
        """
        try:
            data, sr = sf.read(file_path)
            
            if not isinstance(data, np.ndarray):
                raise ValueError(f"Expected numpy array, got {type(data)}")
                
            # Convert to mono if stereo
            if len(data.shape) > 1 and data.shape[1] > 1:
                data = data.mean(axis=1)

            data = data.astype(np.float32)
            if np.max(np.abs(data)) > 0:
                data /= np.max(np.abs(data))

            transcribed_text = self.pipeline({"sampling_rate": sr, "raw": data})["text"]
            logger.info(f"Successfully transcribed audio file: {file_path}")
            return transcribed_text.strip()
        except Exception as e:
            logger.error(f"Error processing audio file {file_path}: {e}")
            return f"Ошибка обработки аудио файла: {str(e)}"


class MessageManager:
    def __init__(self):
        self.queue: int = 0
        self.audio_manager = AudioManager()

    def add_user_message(self, messages: dict, history: Optional[List]):
        """
        Adds a new user message to the conversation history and generates a unique session identifier.

        This function appends the user's input message to the conversation history and increments the
        queue counter to indicate a pending response generation. If history is not provided, a new
        conversation history is initialized.

        :param messages: The user's input message to be added to the conversation history.
        :param history: The existing conversation history as a list of message pairs (user, bot responses).
                        Each pair is a list, with the second item initially set to None for new user messages.
        :return: A tuple containing an empty string (for response text), the updated conversation history,
                 and a unique session ID (uid).
        """
        uid = uuid.uuid4()
        logger.info(f"Processing the question. Queue - {self.queue}. UID - [{uid}]")
        if history is None:
            history = []

        # File processing (audio and images)
        if messages["files"]:
            for file in messages["files"]:
                if isinstance(file, str) and file.endswith('.wav'):  # Audio File
                    # Using AudioManager for transcription
                    transcribed_text = self.audio_manager.transcribe_from_file(file)
                    history.append({"role": "user", "content": transcribed_text})
                else:  # Image
                    history.append({"role": "user", "content": messages["files"]})
                    break

        # Добавляем текстовое сообщение, если оно есть
        if messages["text"]:
            history.append({"role": "user", "content": messages["text"]})

        self.queue += 1
        logger.info(f"The question has been processed. UID - [{uid}]")
        return "", history, uid

    @staticmethod
    def get_recent_history(history: List[dict]) -> List[dict]:
        """
        Gets the last 3 message pairs from the history.

        :param history: Full history of the dialog.
        :return: List with the last message pairs.
        """
        pair_count: int = 0
        temp_history: list = []

        for message in reversed(history[:-1]):  # Exclude the user's last message
            if message["role"] == "user" and pair_count == 0:
                continue

            temp_history.append(message)

            if message["role"] == "assistant":
                pair_count += 1

            if pair_count == 3:
                break

        return list(reversed(temp_history))

    @staticmethod
    def process_retrieved_docs(retrieved_docs: str) -> str:
        """
        Processes the extracted documents by deleting HTML tags.

        :param retrieved_docs: Documents with HTML tags.
        :return: Documents without HTML tags.
        """
        files = re.findall(r'<a\s+[^>]*>(.*?)</a>', retrieved_docs)
        for file in files:
            retrieved_docs = re.sub(fr'<a\s+[^>]*>{file}</a>', file, retrieved_docs)
        return retrieved_docs

    @staticmethod
    def prepare_chat_history(history: List[dict]) -> List[BaseMessage]:
        """
        Converts the dialog history to the Longchain message format.

        :param history: The history of the dialogue in the form of a dictionary list.
        ::return: A list of messages in Long Chain format.
        """
        langchain_messages = []
        for message in history:
            role = message.get("role")
            content = message.get("content", "")

            if role == "user":
                if isinstance(content, tuple):
                    # Processing messages with images
                    human_message = HumanMessage(
                        content=[
                            {"type": "text", "text": langchain_messages[-1].content if langchain_messages else ""},
                            {"type": "image_url", "image_url": {"url": content[0]}}
                        ]
                    )
                    langchain_messages.append(human_message)
                else:
                    langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content or ""))

        return langchain_messages

    def prepare_context_message(self, history: List[dict], retrieved_docs: str, mode: str) -> str:
        """
        Prepares a contextual message based on history and documents.

        :param history: The history of dialogue.
        :param retrieved_docs: Extracted documents.
        :param mode: Operating mode.
        :return: Contextual message.
        """
        last_user_message: str = history[-1].get("content")
        processed_docs = self.process_retrieved_docs(retrieved_docs)

        if processed_docs and mode in MODES:
            last_user_message = (
                f"Контекст: {processed_docs}\n\nИспользуя только контекст, ответь на вопрос: "
                f"{last_user_message}"
            )

        return last_user_message

    @staticmethod
    def add_source_references(
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


class ModelManager:
    def __init__(self, message_manager, prompt_manager, analytics_manager, document_manager=None):
        self.message_manager: MessageManager = message_manager
        self.prompt_manager: SystemPromptManager = prompt_manager
        self.analytics_manager: AnalyticsManager = analytics_manager
        self.document_manager = document_manager

        # Initializing the base model
        self.llm = ClaudeCodeLLM(model_name=CLAUDE_CODE_MODELS[0])

        # Initialize Multi-Agent Manager
        self.multi_agent_manager = None
        if self.document_manager:
            from multi_agent_manager import MultiAgentManager
            self.multi_agent_manager = MultiAgentManager(
                document_manager=self.document_manager,
                model_name=CLAUDE_CODE_MODELS[0]
            )

    def update_model(self, model_name: str):
        """
        Updates the LLM model only if the model has changed.
        
        :param model_name: The name of the model to use.
        :return: None.
        """
        if model_name != self.llm.model_name:
            self.llm = ClaudeCodeLLM(model_name=model_name)
            # Обновляем модель в многоагентной системе
            if self.multi_agent_manager:
                self.multi_agent_manager.update_model(model_name)

    async def generate_response_stream(
        self,
        model: str,
        history: List[dict],
        mode: str,
        retrieved_docs: str,
        scores: List[float],
        is_use_tools: bool,
        uid: str
    ) -> AsyncGenerator[list[dict], None]:
        """
        Generates a response using multi-agent system or traditional approach.

        :param model: Model of the list.
        :param history: List of conversation pairs (user input and bot responses).
        :param mode: Operation mode, which influences the use of context in responses.
        :param retrieved_docs: Relevant documents retrieved to help answer the user query.
        :param scores: List of scores associated with retrieved documents for source filtering.
        :param is_use_tools: A boolean indicating whether to enable multi-agent system.
        :param uid: Unique identifier for the current user session, useful for logging and tracking.
        :return: Yields updated conversation history after each token generated, and the final response with sources.
        """
        logger.info(f"Preparing to generate a response based on context and history [uid - {uid}]")
        if not history or not history[-1].get("role"):
            yield history[:-1]
            return

        logger.info(f"Beginning response generation [uid - {uid}]")

        # Get the last user message
        last_user_message = history[-1].get("content", "") if history else ""

        # Use log-based approach
        logger.info(f"Using log-based RAG approach [uid - {uid}]")
        response_text, _ = await self._log_based_approach(
            history, mode, retrieved_docs, uid, model, is_use_tools
        )

        logger.info(f"Response generation completed [uid - {uid}]")
        history.append({"role": "assistant", "content": response_text})
        yield history
        
        self.message_manager.queue -= 1
        _ = self.analytics_manager.update_message_analytics(history)

    async def _log_based_approach(self, history: List[dict], mode: str, retrieved_docs: str, uid: str, model: str, use_log_search: bool) -> tuple[str, list]:
        """Use MultiAgentManager for hybrid RAG+Logs approach"""
        
        # Update model if changed
        self.update_model(model)
        
        # Get the last user message
        last_user_message = history[-1].get("content", "") if history else ""
        
        # Use Multi-Agent System if logs are enabled and available
        if use_log_search and self.multi_agent_manager and self.document_manager.log_entries:
            try:
                logger.info(f"Using multi-agent system (RAG+Logs) with model {model} [uid - {uid}]")
                # Update model in multi-agent system
                self.multi_agent_manager.update_model(model)
                
                result = self.multi_agent_manager.process_query_sync(
                    query=last_user_message,
                    dialog_history=history,
                    thread_id=uid
                )
                
                response_text = result["answer"]
                files = result["sources"]
                
                # Add information about query type
                query_type_info = ""
                if result["query_type"] == "hybrid":
                    query_type_info = "\n\n*Использованы данные из документов и логов*"
                
                response_text += query_type_info
                
                logger.info(f"Multi-agent response completed: type={result['query_type']} [uid - {uid}]")
                
                return response_text, files
                
            except Exception as e:
                logger.error(f"Error in multi-agent system, falling back to simple approach: {e}")
        
        # Fallback: simple approach without multi-agent system
        logger.info(f"Using simple approach [uid - {uid}]")
        
        files = re.findall(r'<a\s+[^>]*>(.*?)</a>', retrieved_docs)
        recent_history = self.message_manager.get_recent_history(history)
        langchain_messages = self.message_manager.prepare_chat_history(recent_history)
        last_message = self.message_manager.prepare_context_message(history, retrieved_docs, mode)
        langchain_messages.append(HumanMessage(content=last_message))
        
        # Simple hybrid context (fallback)
        context_parts = []
        
        # Get RAG context
        rag_context, rag_sources = self.document_manager.get_rag_context(
            last_user_message, 
            k_documents=8
        )
        if rag_context:
            context_parts.append(f"Контекст из документов:\n{rag_context}")
            files.extend(rag_sources)
        
        # Add logs if enabled
        if use_log_search and self.document_manager.log_entries:
            # Ограничение логов для избежания ошибки "Argument list too long"
            max_logs_chars = 50000  # Максимальный размер логов в символах
            max_log_entries = 1000   # Максимальное количество записей логов
            
            # Берем последние записи и ограничиваем по размеру
            log_entries = self.document_manager.log_entries[-max_log_entries:]
            all_logs = "\n".join(log_entries)
            
            # Если логи все еще слишком большие, обрезаем по символам
            if len(all_logs) > max_logs_chars:
                all_logs = all_logs[-max_logs_chars:]
                # Убеждаемся, что не обрезали строку посередине
                first_newline = all_logs.find('\n')
                if first_newline > 0:
                    all_logs = all_logs[first_newline + 1:]
            
            context_parts.append(f"Последние логи системы ({len(log_entries)} записей):\n{all_logs}")
            files.append("logs")
        
        # Combine context
        if context_parts:
            full_context = "\n\n" + "\n\n".join(context_parts)
            enhanced_message = f"{last_message}{full_context}"
            langchain_messages[-1] = HumanMessage(content=enhanced_message)
        
        # Generate response
        try:
            response = self.llm.invoke(langchain_messages)
            response_text = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            response_text = f"Ошибка при генерации ответа: {str(e)}"

        return response_text, files



class UIManager:
    def __init__(self):
        self.message_manager: MessageManager = MessageManager()
        self.prompt_manager: SystemPromptManager = SystemPromptManager()
        self.analytics_manager: AnalyticsManager = AnalyticsManager()
        self.audio_manager: AudioManager = AudioManager()
        self.document_manager: DocumentManager = DocumentManager()
        self.vm_manager: VMManager = VMManager()
        self.auth_manager: AuthManager = AuthManager(self.document_manager)
        
        # Initialize ModelManager with document_manager for multi-agent system
        self.model_manager: ModelManager = ModelManager(
            self.message_manager, 
            self.prompt_manager, 
            self.analytics_manager,
            self.document_manager  # Pass document_manager for multi-agent system
        )
        
        # Initialize Neo4j directories for Docker volumes
        self._create_neo4j_directories()
        
        # Текущая сессия диалога
        self.current_session_id: str = str(uuid.uuid4())
    
    def _create_neo4j_directories(self):
        """
        Create Neo4j directories for Docker volumes.
        """
        try:
            base_path = "/home/timur/PycharmWork/LocalGPT"
            neo4j_dirs = [
                f"{base_path}/neo4j/data",
                f"{base_path}/neo4j/logs", 
                f"{base_path}/neo4j/import",
                f"{base_path}/neo4j/plugins"
            ]
            
            for dir_path in neo4j_dirs:
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"Created Neo4j directory: {dir_path}")
                
        except Exception as e:
            logger.error(f"Error creating Neo4j directories: {e}")

    def create_new_dialog(self) -> tuple:
        """
        Создает новый диалог.
        """
        self.current_session_id = str(uuid.uuid4())
        return [], self.get_dialog_choices()
    
    def get_dialog_choices(self) -> gr.update:
        """
        Получает список сохраненных диалогов для UI.
        """
        dialogs = self.analytics_manager.get_all_dialogs()
        choices = []
        for dialog in dialogs:
            # Укорачиваем заголовок для лучшего отображения
            title = dialog['title']
            if len(title) > 40:
                title = title[:40] + "..."
            date_str = dialog['updated_at'][:16].replace('T', ' ')
            display_name = f"💬 {title}"
            choices.append((display_name, dialog['session_id']))
        return gr.update(choices=choices)
    
    def get_dialog_radio_choices(self):
        """Получить список диалогов для RadioGroup"""
        dialogs = self.analytics_manager.get_all_dialogs()
        choices = []
        for dialog in dialogs[:15]:  # Ограничиваем количество для удобства
            title = dialog['title']
            if len(title) > 35:
                title = title[:35] + "..."
            date_str = dialog['updated_at'][:16].replace('T', ' ')
            display_name = f"💬 {title} ({date_str})"
            choices.append((display_name, dialog['session_id']))
        return gr.update(choices=choices, value=None)
    
    def get_dialog_list_html(self) -> str:
        """Создает HTML-список диалогов в стиле ChatGPT."""
        dialogs = self.analytics_manager.get_all_dialogs()
        if not dialogs:
            return """
            <div class="empty-dialogs">
                <span style="font-size: 48px; display: block; margin-bottom: 12px;">💬</span>
                <div>Нет сохраненных диалогов</div>
                <div style="font-size: 12px; margin-top: 8px; opacity: 0.6;">Начните новый диалог, чтобы он появился здесь</div>
            </div>
            """
        
        html_items = []
        for dialog in dialogs[:20]:  # Показываем только последние 20
            title = dialog['title']
            if len(title) > 40:
                title = title[:40] + "..."
            
            date_str = dialog['updated_at'][:16].replace('T', ' ')
            session_id = dialog['session_id']
            
            # Проверяем, является ли этот диалог активным
            is_selected = session_id == self.current_session_id
            selected_class = " selected" if is_selected else ""
            
            html_items.append(f"""
            <div class="dialog-item{selected_class}" 
                 data-session-id="{session_id}" 
                 onclick="loadDialogById('{session_id}')">
                <div class="dialog-item-title">
                    <span>💬</span>
                    <span>{title}</span>
                </div>
                <div class="dialog-item-date">
                    {date_str}
                </div>
            </div>
            """)
        
        return f"""
        <div class="dialog-list-container">
            {''.join(html_items)}
        </div>
        <script>
        if (typeof loadDialogById === 'undefined') {{
            window.loadDialogById = function(sessionId) {{
                console.log('Loading dialog:', sessionId);
                
                // Обновляем визуальное выделение
                document.querySelectorAll('.dialog-item').forEach(item => {{
                    item.classList.remove('selected');
                }});
                
                const selected = document.querySelector('[data-session-id="' + sessionId + '"]');
                if (selected) {{
                    selected.classList.add('selected');
                }}
                
                // Находим скрытый текстбокс и устанавливаем значение
                const textInput = document.querySelector('#selected-dialog-id input') ||
                                 document.querySelector('#selected-dialog-id textarea');
                
                if (textInput) {{
                    textInput.value = sessionId;
                    textInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    console.log('Dialog ID set to:', sessionId);
                }} else {{
                    console.error('Hidden textbox not found');
                }}
                
                // Находим и кликаем по скрытой кнопке
                const loadBtn = document.getElementById('load-dialog-btn');
                if (loadBtn) {{
                    loadBtn.click();
                    console.log('Load button clicked');
                }} else {{
                    console.error('Load button not found');
                }}
            }};
        }}
        </script>
        """
    
    
    def load_selected_dialog(self, selected_dialog_id: str) -> tuple:
        """
        Загружает выбранный диалог.
        """
        if not selected_dialog_id:
            return [], self.current_session_id
            
        messages = self.analytics_manager.load_dialog_session(selected_dialog_id)
        if messages:
            self.current_session_id = selected_dialog_id
            return messages, selected_dialog_id
        return [], self.current_session_id
    
    def delete_selected_dialog(self, selected_dialog_id: str) -> tuple:
        """
        Удаляет выбранный диалог.
        """
        if selected_dialog_id and self.analytics_manager.delete_dialog_session(selected_dialog_id):
            gr.Info("Диалог удален")
            # Если удаляем текущий диалог, создаем новый
            if selected_dialog_id == self.current_session_id:
                self.current_session_id = str(uuid.uuid4())
                return [], None, self.get_dialog_choices()
        return [], selected_dialog_id, self.get_dialog_choices()
    
    def auto_save_dialog(self, messages: List[dict]):
        """
        Автоматически сохраняет диалог после каждого сообщения.
        """
        if messages and len(messages) > 0:
            self.analytics_manager.save_dialog_session(self.current_session_id, messages)
    
    @staticmethod
    def update_chat_label(selected_model: str) -> tuple:
        """
        Updates the label of the chat interface based on the selected model.

        :param: selected_model (str): The name of the currently selected model.

        :returns: A tuple of two Gradio updates. The first update sets the label of the chat
                  interface to the name of the selected model. The second update sets the interactive status
                  of the chat interface to True if the selected model is not the "llm" model, or False otherwise.
        """
        return gr.update(label=f"LLM: {selected_model}"), gr.update(interactive=True)

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

        :return: gr.Blocks: The Gradio Blocks instance for the LocalGPT application UI.
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
            current_session = gr.State(self.current_session_id)

            with gr.Tab("Чат"):
                with gr.Row():
                    # Боковая панель для истории диалогов
                    with gr.Column(scale=1, elem_classes=["sidebar-container"]):
                        with gr.Row(equal_height=True):
                            with gr.Column():
                                # Заголовок сайдбара
                                gr.HTML("""<h2><center>💬 История диалогов</center></h2>""")
                                
                                # Кнопка нового диалога
                                new_dialog_btn = gr.Button(
                                    f"{CHATGPT_ICONS['new_chat']} Новый чат", 
                                    variant="primary"
                                )
                                
                                
                                # RadioGroup список диалогов
                                dialog_radio = gr.Radio(
                                    choices=[],
                                    value=None,
                                    show_label=False,
                                    interactive=True,
                                    container=False,
                                    elem_classes=["dialog-radio"]
                                )
                                
                                # Кнопки действий
                                delete_dialog_btn = gr.Button(
                                    f"{CHATGPT_ICONS['delete']} Удалить", 
                                    variant="secondary",
                                    elem_classes=["delete-btn"]
                                )
                    
                    # Основная область чата
                    with gr.Column(scale=3):
                        with gr.Row(equal_height=True):
                            with gr.Column():
                                collection_radio = gr.Radio(
                                    choices=MODES,
                                    value=self.prompt_manager.mode,
                                    show_label=False
                                )
                                is_use_tools = gr.Checkbox(label="Использовать логи (+ RAG документы)", value=True)

                            with gr.Column():
                                model = gr.Dropdown(
                                    choices=CLAUDE_CODE_MODELS,
                                    value=CLAUDE_CODE_MODELS[0],
                                    interactive=True,
                                    show_label=True,
                                    label="Выбор моделей Claude"
                                )

                        with gr.Row():
                            with gr.Column(scale=10):
                                chatbot = gr.Chatbot(
                                    label=f"LLM: {model.value}",
                                    height=500,
                                    type="messages",
                                    show_copy_button=True,
                                    avatar_images=(
                                        AVATAR_USER,
                                        AVATAR_BOT
                                    )
                                )

                        with gr.Row(equal_height=True):
                            with gr.Column(scale=10):
                                msg = gr.MultimodalTextbox(
                                    label="Отправить сообщение",
                                    placeholder="👉 Напишите запрос",
                                    sources=["upload", "microphone"],
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
                        with gr.Tab("Документы"):
                            upload_files = gr.Files(
                                label="Загрузка документов",
                                file_count="multiple"
                            )
                            file_warning = gr.Markdown("Фрагменты ещё не загружены!")
                        
                        with gr.Tab("Логи"):
                            log_file_upload = gr.File(
                                label="Загрузить лог файл (.txt)",
                                file_types=[".txt"]
                            )
                            log_status = gr.Markdown("Лог файл не загружен")
                        
                        with gr.Tab("CSV Логи из data2"):
                            with gr.Row():
                                user_request_time = gr.DateTime(
                                    label="Время запроса пользователя",
                                    value=None,
                                    info="Будут показаны логи от этого времени до 2 дней назад"
                                )
                                user_pid = gr.Textbox(
                                    label="PID пользователя",
                                    placeholder="Например: p1, p2, p3...",
                                    value=""
                                )
                            load_csv_logs_btn = gr.Button("📊 Загрузить CSV логи", variant="primary")
                            csv_log_status = gr.Markdown("CSV логи не загружены")

                    with gr.Column(scale=7):
                        files_selected = gr.Dropdown(
                            choices=None,
                            label="Выберите файлы для удаления",
                            value="",
                            multiselect=True
                        )
                        delete = gr.Button("🧹 Удалить", variant="primary", elem_classes=["delete-btn"])

            with gr.Tab("Настройки", visible=False) as settings_tab:
                with gr.Column():
                    with gr.Row():
                        status_output = gr.Textbox(label="Текущий статус сервера", interactive=False)
                        action_dropdown = gr.Dropdown(
                            choices=[
                                "Статус", "Вкл", "Выкл", "Перезагрузить",
                                "Архивировать", "Разархивировать", "Авторизация"
                            ],
                            value="Выберите действие",
                            allow_custom_value=True,
                            label="Выберите операцию с сервером",
                            interactive=True,
                        )
                        action_dropdown.change(
                            fn=self.vm_manager.control_vm,
                            inputs=action_dropdown,
                            outputs=status_output
                        )

                with gr.Accordion("Параметры", open=False):
                    with gr.Tab(label="Параметры извлечения фрагментов из текста"):
                        k_documents = gr.Slider(
                            minimum=1,
                            maximum=12,
                            value=8,
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
                            show_search="search",
                            show_row_numbers=True,
                            show_fullscreen_button=True,
                            show_copy_button=True,
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
                outputs=[
                    local_data,
                    documents_tab,
                    settings_tab,
                    logging_tab,
                    login_btn,
                    modal,
                    message_login,
                    files_selected
                ]
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
                js="() => { removeStorage('access_token'); return null; }"
            )

            cancel_login.click(
                fn=lambda: Modal(visible=False),
                inputs=None,
                outputs=modal
            )

            model.change(
                fn=self.update_chat_label,
                inputs=[model],
                outputs=[chatbot, is_use_tools],
                js=JS_MODEL_TOGGLE
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

            # Upload log file
            log_file_upload.upload(
                fn=self.document_manager.load_log_file_ui,
                inputs=[log_file_upload],
                outputs=[log_status],
                queue=True
            )

            # Load CSV logs from data2
            load_csv_logs_btn.click(
                fn=self.document_manager.load_csv_logs_from_data,
                inputs=[user_request_time, user_pid],
                outputs=[csv_log_status],
                queue=True
            )

            # Delete documents from db
            delete.click(
                fn=self.document_manager.delete_documents,
                inputs=files_selected,
                outputs=[files_selected]
            )

            # Обновление списка диалогов при загрузке
            demo.load(
                fn=self.get_dialog_radio_choices,
                outputs=dialog_radio
            )
            
            # Новый диалог
            def new_dialog_with_update():
                chat, _ = self.create_new_dialog()
                return chat, str(uuid.uuid4())
                
            new_dialog_btn.click(
                fn=new_dialog_with_update,
                outputs=[chatbot, current_session]
            ).success(
                fn=self.get_dialog_radio_choices,
                outputs=dialog_radio
            )
            
            # Выбор диалога из radio  
            dialog_radio.change(
                fn=self.load_selected_dialog,
                inputs=dialog_radio,
                outputs=[chatbot, current_session]
            )
            
            
            # Удаление диалога
            def delete_dialog_with_update(selected_id):
                chat, _, _ = self.delete_selected_dialog(selected_id)
                return chat, str(uuid.uuid4())
                
            delete_dialog_btn.click(
                fn=delete_dialog_with_update,
                inputs=dialog_radio,
                outputs=[chatbot, current_session]
            ).success(
                fn=self.get_dialog_radio_choices,
                outputs=dialog_radio
            )

            # Pressing Enter
            click_msg_event = msg.submit(
                fn=self.message_manager.add_user_message,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot, uid],
                queue=False,
            ).success(
                fn=self.document_manager.retrieve_documents,
                inputs=[chatbot, collection_radio, k_documents, uid],
                outputs=[retrieved_docs, scores],
                queue=True,
            ).success(
                fn=self.model_manager.generate_response_stream,
                inputs=[model, chatbot, collection_radio, retrieved_docs, scores, is_use_tools, uid],
                outputs=chatbot,
                queue=True
            ).success(
                fn=self.auto_save_dialog,
                inputs=chatbot,
                queue=False
            ).success(
                fn=self.get_dialog_radio_choices,
                outputs=dialog_radio
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
                fn=new_dialog_with_update,
                outputs=[chatbot, current_session],
                queue=False,
                js=JS
            ).success(
                fn=self.get_dialog_radio_choices,
                outputs=dialog_radio
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
                inputs=[local_data, gr.State(False)],
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
                js=f"{LOCAL_STORAGE}; {JS_CHATGPT_STYLE}();"
            )

        demo.queue(max_size=128, api_open=False, default_concurrency_limit=5)
        return demo
