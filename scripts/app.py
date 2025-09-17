import uuid
import glob
import os.path
import tempfile
import numpy as np
import pandas as pd
import gradio as gr
import soundfile as sf
from __init__ import *
from gradio_modal import Modal
from neo4j import GraphDatabase
from tinydb import TinyDB, where
from functions.functions import *
from datetime import datetime, timedelta
from claude_code_llm import ClaudeCodeLLM
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Optional, Tuple, AsyncGenerator, Union
from langchain_neo4j import Neo4jVector, Neo4jGraph, GraphCypherQAChain
from langchain_experimental.graph_transformers import LLMGraphTransformer
from natasha import MorphVocab, Segmenter, NewsMorphTagger, NewsEmbedding


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
        # Отдельная база для истории диалогов
        self.dialogs_db: TinyDB = TinyDB(f'{QUESTIONS}/dialogs.json', indent=4, ensure_ascii=False)

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

        obj_tabs: List[Union[gr.update, None, str]] = [local_data] + [gr.update(visible=is_logged_in) for _ in range(2)]
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

        obj_tabs = [gr.update(visible=not is_logged_in)] + [gr.update(visible=False) for _ in range(2)]
        obj_tabs.append(gr.update(value="Войти", icon=LOGIN_ICON if is_logged_in else login_btn))

        return obj_tabs


class DocumentManager:
    def __init__(self):
        # self.embeddings: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
        #     model_name=EMBEDDER_NAME,
        #     cache_folder=MODELS_DIR
        # )
        self.embeddings: OpenAIEmbeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.collection: str = "all-documents"
        
        # Neo4j connection parameters
        self.neo4j_url = "bolt://localhost:7687"
        self.neo4j_username = "neo4j"
        self.neo4j_password = "localgpt123"
        
        # Initialize Neo4j Graph for structured queries
        self.neo4j_graph = Neo4jGraph(
            url=self.neo4j_url,
            username=self.neo4j_username,
            password=self.neo4j_password
        )
        
        # Initialize Neo4j connections - проверяем существующий индекс
        self._try_initialize_existing_vector_index()
        
        # Initialize Graph Database driver
        self.graph_driver = GraphDatabase.driver(
            self.neo4j_url,
            auth=(self.neo4j_username, self.neo4j_password)
        )
        
        # Other components
        self.log_entries: List[str] = []  # Cached log entries
        self.csv_logs_data: pd.DataFrame = pd.DataFrame()  # CSV logs data
        self.data_path: str = "/home/timur/PycharmWork/LocalGPT/logs"  # Path to data folder
        
        # Initialize LLM for graph construction
        self.llm = ChatOpenAI(model_name="gpt-4.1", temperature=0)
        self.llm_transformer = LLMGraphTransformer(llm=self.llm)
        self.cypher_llm = ChatOpenAI(temperature=0)
    
    def _try_initialize_existing_vector_index(self):
        """
        Попытка подключиться к существующему векторному индексу при запуске.
        """
        try:
            # Проверяем, есть ли существующий векторный индекс
            self.neo4j_vector: Optional[Neo4jVector] = Neo4jVector.from_existing_index(
                embedding=self.embeddings,
                index_name="vector",
                url=self.neo4j_url,
                username=self.neo4j_username,
                password=self.neo4j_password,
                node_label="Document",
                text_node_property="text",
                embedding_node_property="embedding",
                search_type="hybrid",
                keyword_index_name="keyword"
            )
            logger.info("✅ Подключились к существующему векторному индексу")
        except Exception as e:
            logger.info(f"ℹ️ Существующий векторный индекс не найден (это нормально при первом запуске): {e}")
            self.neo4j_vector = None

    def _normalize_neo4j_label(self, label: str) -> str:
        """
        Нормализация лейбла для Neo4j (убирает пробелы и спец. символы)
        
        :param label: Исходный лейбл
        :return: Нормализованный лейбл
        """
        if not label:
            return "Entity"
        
        # Заменяем пробелы, дефисы и другие символы на подчёркивания
        normalized = label.replace(" ", "_").replace("-", "_").replace("'", "").replace(".", "_")
        
        # Удаляем недопустимые символы и оставляем только буквы, цифры и подчёркивания
        normalized = "".join(c for c in normalized if c.isalnum() or c == "_")
        
        # Убираем начальные цифры (Neo4j лейблы не могут начинаться с цифры)
        while normalized and normalized[0].isdigit():
            normalized = normalized[1:]
        
        # Если после очистки лейбл пустой, возвращаем дефолтный
        if not normalized:
            return "Entity"
        
        return normalized

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
        lines = [line.strip() for line in lines if len(line.strip()) > 2]
        page_content = " ".join(lines).strip()
        return "" if len(page_content) < 10 else page_content
    
    def _create_llm_graph_relationships(self, documents: List[Document], ids: List[str]):
        """
        Create relationships using LLM Graph Transformer и добавляем эмбеддинги для чанков!
        """
        try:
            logger.info(f"🕸️ Создаём граф знаний для {len(documents)} документов с помощью LLM...")
            
            # Используем LLM для извлечения графа из каждого документа
            graph_documents = self.llm_transformer.convert_to_graph_documents(documents)
            self.neo4j_graph.add_graph_documents(graph_documents, include_source=True)
            
            logger.info("🎉 LLM граф знаний создан успешно!")
            
            # Добавляем NEXT_CHUNK связи между последовательными чанками
            logger.info("🔗 Создаём связи NEXT_CHUNK между чанками...")
            with self.graph_driver.session() as session:
                for i, (doc, doc_id) in enumerate(zip(documents, ids)):
                    # Устанавливаем ID для чанка
                    session.run("""
                        MATCH (d:Document {text: $text}) 
                        SET d.chunk_id = $chunk_id,
                            d.image = $image
                        """,
                        text=doc.page_content,
                        chunk_id=doc_id,
                        image="📄 " + doc.page_content[:20] + "..."
                    )
                    
                    # Связываем с предыдущим чанком если это не первый
                    if i > 0:
                        prev_doc_id = ids[i-1]
                        session.run(
                            """
                                MATCH (prev:Document {chunk_id: $prev_id})
                                MATCH (curr:Document {chunk_id: $curr_id})
                                MERGE (prev)-[:NEXT_CHUNK]->(curr)
                            """, 
                            prev_id=prev_doc_id, curr_id=doc_id
                        )
            
            logger.info("✅ NEXT_CHUNK связи созданы!")
            
            # Создаём векторный индекс с эмбеддингами для тех же документов
            logger.info(f"🔗 Создаём векторные эмбеддинги для {len(documents)} чанков...")
            
            # Создаём Neo4jVector с эмбеддингами
            self.neo4j_vector = Neo4jVector.from_existing_graph(
                embedding=self.embeddings,
                node_label="Document",
                embedding_node_property="embedding",
                text_node_properties=["text"],
                url=self.neo4j_url,
                username=self.neo4j_username,
                password=self.neo4j_password,
                index_name="vector",
                search_type="hybrid"
            )
            
            logger.info("🎯 Векторные эмбеддинги созданы и связаны с графом!")
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания LLM графа или эмбеддингов: {e}")
    
    
    def update_documents(self, fixed_documents: List[Document], ids: List[str]) -> tuple[bool, str]:
        """
        Updates existing documents in the database (Neo4j or Chroma fallback).
        """
        try:
            # Neo4j approach
            # Check for existing documents and remove duplicates
            existing_docs = self._get_existing_document_names()
            new_files = {os.path.basename(doc.metadata["source"]) for doc in fixed_documents}
            
            if same_files := new_files & existing_docs:
                gr.Warning("Файлы " + ", ".join(same_files) + " повторяются, поэтому они будут обновлены")
                self._delete_documents_by_names(list(same_files))
            
            # Create graph relationships
            self._create_llm_graph_relationships(fixed_documents, ids)
            
            file_warning = f"Загружено {len(fixed_documents)} фрагментов в Neo4j GraphRAG! Можно задавать вопросы."
            return True, file_warning
            
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
            if self.neo4j_graph:
                # Neo4j approach
                self._create_llm_graph_relationships(fixed_documents, ids)
                file_warning = f"Загружено {len(fixed_documents)} фрагментов в Neo4j GraphRAG! Можно задавать вопросы."
            
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
    
    def _get_existing_document_names(self) -> set:
        """
        Get existing document names from Neo4j.
        """
        if not self.graph_driver:
            return set()
        
        try:
            with self.graph_driver.session() as session:
                result = session.run("MATCH (d:Document) RETURN DISTINCT d.source as source")
                return {record["source"] for record in result if record["source"]}
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
                        "MATCH (d:Document {source: $source}) DETACH DELETE d",
                        source=filename
                    )
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")

    def retrieve_documents(
        self,
        history: List[dict],
        collection_radio: str,
        k_documents: int
    ) -> str:
        """
        UI wrapper for get_rag_context to maintain UI compatibility.
        
        :param history: The conversation history
        :param collection_radio: The selected collection mode
        :param k_documents: The number of documents to retrieve  
        :return: Formatted documents and scores for UI display
        """
        if (
            collection_radio not in MODES
            or not history
            or history[-1]["role"] != "user"
        ):
            return "Появятся после задавания вопросов", []

        last_user_message = history[-1].get("content")
        docs = []
        
        try:
            # Get Cypher query and graph context
            graph_context, cypher_query = self._get_graph_context(last_user_message)
            
            # Use documents with custom retrieval query if we have a cypher query
            if cypher_query:
                docs = self._search_with_custom_cypher(last_user_message, k_documents, cypher_query)
            
            if not docs:
                docs = self._search_with_custom_cypher(last_user_message, k_documents, cypher_query='')
            
            # Format docs for UI display
            formatted_docs = []
            for doc, score in docs:
                source = doc.metadata.get("source", "")
                url = f'<a href="file/{source}" target="_blank" rel="noopener noreferrer">{os.path.basename(source)}</a>'
                
                # Format deeper connections
                deeper_connections = doc.metadata.get("deeper_connections", [])
                connections_html = ""
                if deeper_connections:
                    connections_html = "<br><strong>Graph Relations:</strong><br>"
                    connections_html += "<small>• = связанные узлы (тип связи), ↳ = узлы второго уровня (тип связи)</small><br>"
                    for conn in deeper_connections:
                        level1_node = conn.get("level1_node", {})
                        level1_rel = conn.get("level1_relationship_type", "")
                        level2_connections = conn.get("level2_connections", [])
                        
                        connections_html += f"• {level1_node.get('id', 'Unknown')} <em>({level1_rel})</em><br>"
                        for l2_conn in level2_connections:
                            l2_node = l2_conn.get("level2_node", {})
                            l2_rel = l2_conn.get("level2_relationship_type", "")
                            connections_html += f"  ↳ {l2_node.get('id', 'Unknown')} <em>({l2_rel})</em><br>"
                
                document_html = f"""
                <div style="border: 1px solid #ddd; margin: 10px 0; padding: 10px; border-radius: 5px;">
                    <strong>Document:</strong> {url}<br>
                    <strong>Score:</strong> {round(score, 3)}<br>
                    <strong>Text:</strong> {doc.page_content}...<br>
                    {connections_html}
                </div>
                """
                formatted_docs.append(document_html)
            
            result_html = "".join(formatted_docs)
            if graph_context:
                result_html = result_html + f"""<br>
                <div>
                    <strong>Graph Cypher Query:</strong>
                    <br>{cypher_query}</br>
                </div>
                <div>
                    <strong>Graph Context:</strong>
                    <br>{graph_context}</br>
                </div>
                """
            
            return result_html
        except Exception as e:
            logger.error(f"Error retrieving documents for UI: {e}")
            return f"Ошибка при поиске документов: {str(e)}", []
    
    def _get_graph_context(self, query: str) -> Tuple[str, str]:
        """
        Get additional context from Neo4j graph using GraphCypherQAChain.
        Returns tuple of (context, cypher_query)
        """
        if not self.neo4j_graph or not self.llm:
            return "", ""
        
        try:
            # # Создаём GraphCypherQAChain с return_intermediate_steps=True
            # cypher_qa = GraphCypherQAChain.from_llm(
            #     graph=self.neo4j_graph, 
            #     llm=self.llm,
            #     cypher_llm=self.cypher_llm,
            #     allow_dangerous_requests=True,
            #     verbose=True,
            #     return_direct=True,
            #     return_intermediate_steps=True
            # )
            
            # # Запрашиваем контекст из графа знаний
            # logger.info(f"🔍 Поиск в графе знаний: {query}")
            # result = cypher_qa(query)
            
            # # Теперь result содержит и промежуточные шаги
            # cypher_query = result.get("intermediate_steps", [{}])[-1].get("query")
            # answer = result.get("result", "")
            
            # logger.info(f"🔍 Сгенерированный Cypher запрос: {cypher_query}")
            # logger.info(f"📊 Найден контекст из графа: {len(answer)} символов")

            # return answer, cypher_query
            
            return "", ""
        except Exception as e:
            logger.error(f"❌ Ошибка GraphCypherQAChain: {e}")
            return "", ""

    def _search_with_custom_cypher(self, query: str, k: int, cypher_query: str):
        """
        Perform document search using a custom Cypher query combined with vector similarity.
        """
        try:            
            # Устанавливаем кастомный retrieval_query для поиска связанных узлов
            self.neo4j_vector.retrieval_query = RETRIEVAL_QUERY
            
            docs = self.neo4j_vector.similarity_search_with_score(query, k=k)
            logger.info(f"🔍 Custom search returned {len(docs)} documents")
            
            return docs
            
        except Exception as e:
            logger.error(f"❌ Error in custom Cypher search: {e}")
            return []

    def list_ingested_documents(self):
        """
        Retrieves a list of ingested document filenames from Neo4j or Chroma.

        :return: An update object for UI elements with the current list of ingested document filenames.
        """
        try:
            files = set()
            
            if self.neo4j_graph and self.graph_driver:
                # Get files from Neo4j
                with self.graph_driver.session() as session:
                    result = session.run("MATCH (d:Document) RETURN DISTINCT d.source as source")
                    files = {os.path.basename(record["source"]) for record in result if record["source"]}
            
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
            if self.neo4j_graph and self.graph_driver:
                # Delete from Neo4j
                self._delete_documents_by_names(documents)
                # Also delete from vector store if possible
                # Note: Neo4jVector doesn't have a direct delete by filename method
                # This would require custom implementation
            
            return self.list_ingested_documents()
            
        except Exception as e:
            logger.error(f"Error during document deletion: {e}")
            return gr.update(choices=[])
    
    def load_csv_logs_from_data(self, request_time: str, match_id: str) -> str:
        """
        Load and filter CSV logs from logs folder based on time range and key fields.
        
        :param request_time: User request time in ISO format or empty string
        :param match_id: Match ID filter or empty string
        :return: Status message
        """
        try:
            if not os.path.exists(self.data_path):
                return f"❌ Папка {self.data_path} не найдена"
            
            # Find all CSV files in logs folder
            csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
            if not csv_files:
                return "❌ CSV файлы не найдены в папке logs"
            
            # Load and combine all CSV files with different structures
            all_data = {}
            total_rows = 0
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    file_name = os.path.basename(csv_file)
                    all_data[file_name] = df
                    total_rows += len(df)
                    logger.info(f"Загружен файл {file_name}: {len(df)} записей")
                except Exception as e:
                    logger.warning(f"Не удалось загрузить {csv_file}: {e}")
                    continue
            
            if not all_data:
                return "❌ Не удалось загрузить ни одного CSV файла"
            
            # Apply filters for key fields
            filter_params = {
                'match_id': match_id.strip() if match_id and match_id.strip() else None
            }
            
            # Remove None values from filter params
            active_filters = {k: v for k, v in filter_params.items() if v is not None}
            
            if active_filters:
                filtered_data = {}
                for file_name, df in all_data.items():
                    temp_df = df.copy()
                    
                    # Apply each filter
                    for field, value in active_filters.items():
                        if field in df.columns:
                            try:
                                # Try to convert value to appropriate type
                                if df[field].dtype in ['int64', 'float64']:
                                    filter_value = int(value)
                                else:
                                    filter_value = value
                                
                                temp_df = temp_df[temp_df[field] == filter_value]
                                
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Ошибка фильтрации по {field} в файле {file_name}: {e}")
                                continue
                    
                    if not temp_df.empty:
                        filtered_data[file_name] = temp_df
                
                if not filtered_data:
                    filter_desc = ", ".join([f"{k}={v}" for k, v in active_filters.items()])
                    return f"❌ Не найдено записей для фильтров: {filter_desc}"
                
                all_data = filtered_data
                total_rows = sum(len(df) for df in all_data.values())
            
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
                    
                    # Convert request_dt to Unix timestamp for comparison with start_time
                    request_unix = int(request_dt.timestamp())
                    two_days_back_unix = int(two_days_back.timestamp())
                    
                    filtered_data = {}
                    for file_name, df in all_data.items():
                        if 'start_time' in df.columns:
                            try:
                                # Filter by Unix timestamp
                                filtered_df = df[
                                    (df['start_time'] >= two_days_back_unix) & 
                                    (df['start_time'] <= request_unix)
                                ]
                                if not filtered_df.empty:
                                    filtered_data[file_name] = filtered_df
                            except Exception as e:
                                logger.warning(f"Ошибка фильтрации по времени в файле {file_name}: {e}")
                                continue
                        else:
                            # Keep files without start_time column
                            filtered_data[file_name] = df
                    
                    if not filtered_data:
                        return f"❌ Не найдено записей в диапазоне от {two_days_back} до {request_dt}"
                    
                    all_data = filtered_data
                    total_rows = sum(len(df) for df in all_data.values())
                    
                except Exception as e:
                    return f"❌ Ошибка парсинга времени запроса: {e}"
            
            # Store the filtered data
            self.csv_logs_data = all_data
            
            # Convert to log entries format for compatibility with existing system
            log_entries = []
            for file_name, df in all_data.items():
                for _, row in df.iterrows():
                    # Create a log entry string with key information
                    log_parts = []
                    log_parts.append(f"File: {file_name}")
                    
                    # Add key identification fields first
                    key_fields = ['match_id', 'account_id', 'hero_id', 'player_slot']
                    for field in key_fields:
                        if field in row.index and pd.notna(row[field]):
                            log_parts.append(f"{field.upper()}: {row[field]}")
                    
                    # Add start_time if present and convert to readable format
                    if 'start_time' in row.index and pd.notna(row['start_time']):
                        try:
                            timestamp = pd.to_datetime(row['start_time'], unit='s')
                            log_parts.append(f"Start_Time: {timestamp}")
                        except:
                            log_parts.append(f"Start_Time: {row['start_time']}")
                    
                    # Add other important columns based on file type
                    important_cols = ['duration', 'radiant_win', 'game_mode', 'kills', 'deaths', 'assists', 'gold', 'xp_per_min']
                    for col in important_cols:
                        if col in row.index and pd.notna(row[col]):
                            log_parts.append(f"{col}: {row[col]}")
                    
                    # Add first few additional columns (limit to avoid too long entries)
                    excluded_cols = key_fields + ['start_time'] + important_cols
                    other_cols = [col for col in row.index 
                                 if col not in excluded_cols 
                                 and pd.notna(row[col])][:3]  # Reduced to 3 for cleaner output
                    for col in other_cols:
                        log_parts.append(f"{col}: {row[col]}")
                    
                    log_entries.append(" | ".join(log_parts))
            
            self.log_entries = log_entries
            
            # Prepare summary
            summary_parts = [
                f"✅ Загружено {total_rows} записей из CSV логов",
                f"📁 Файлов: {len(all_data)}",
            ]
            
            # List loaded files
            file_summary = []
            for file_name, df in all_data.items():
                file_summary.append(f"• {file_name}: {len(df)} записей")
            summary_parts.extend(file_summary)
            
            # Add active filter information
            if active_filters:
                filter_info = []
                for field, value in active_filters.items():
                    filter_info.append(f"{field.replace('_', ' ').title()}: {value}")
                summary_parts.append("🎯 Фильтры: " + ", ".join(filter_info))
            
            if request_time:
                summary_parts.append(f"⏰ Период: 2 дня назад от {pd.to_datetime(request_time, unit='s')}")
            
            # Add unique match count if available
            unique_matches = set()
            for df in all_data.values():
                if 'match_id' in df.columns:
                    unique_matches.update(df['match_id'].dropna().unique())
            
            if unique_matches:
                summary_parts.append(f"🎮 Уникальных матчей: {len(unique_matches)}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error loading CSV logs: {e}")
            return f"❌ Ошибка загрузки CSV логов: {str(e)}"
    
    def get_csv_logs_summary(self) -> dict:
        """
        Get summary information about loaded CSV logs.
        
        :return: Dictionary with summary information
        """
        if not self.csv_logs_data or (isinstance(self.csv_logs_data, dict) and not self.csv_logs_data):
            return {"status": "no_data", "message": "CSV логи не загружены"}
        
        try:
            # Handle new format where csv_logs_data is a dict of DataFrames
            if isinstance(self.csv_logs_data, dict):
                total_records = sum(len(df) for df in self.csv_logs_data.values())
                unique_matches = set()
                time_range = {"start": "N/A", "end": "N/A"}
                source_files = list(self.csv_logs_data.keys())
                
                # Collect unique match IDs and time range
                start_times = []
                for file_name, df in self.csv_logs_data.items():
                    if 'match_id' in df.columns:
                        unique_matches.update(df['match_id'].dropna().unique())
                    
                    if 'start_time' in df.columns:
                        start_times.extend(df['start_time'].dropna().tolist())
                
                # Calculate time range if we have start_times
                if start_times:
                    min_time = pd.to_datetime(min(start_times), unit='s')
                    max_time = pd.to_datetime(max(start_times), unit='s')
                    time_range = {
                        "start": min_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "end": max_time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                
                summary = {
                    "status": "loaded",
                    "total_records": total_records,
                    "unique_matches": len(unique_matches),
                    "unique_files": len(source_files),
                    "time_range": time_range,
                    "source_files": source_files
                }
            else:
                # Fallback for old format
                df = self.csv_logs_data
                summary = {
                    "status": "loaded",
                    "total_records": len(df),
                    "unique_matches": df['match_id'].nunique() if 'match_id' in df.columns else 0,
                    "time_range": {
                        "start": df['start_time'].min() if 'start_time' in df.columns and not df['start_time'].isna().all() else "N/A",
                        "end": df['start_time'].max() if 'start_time' in df.columns and not df['start_time'].isna().all() else "N/A"
                    },
                    "source_files": [df.get('source_file', 'Unknown')]
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
        is_use_tools: bool,
        uid: str
    ) -> AsyncGenerator[list[dict], None]:
        """
        Generates a response using multi-agent system or traditional approach.

        :param model: Model of the list.
        :param history: List of conversation pairs (user input and bot responses).
        :param mode: Operation mode, which influences the use of context in responses.
        :param retrieved_docs: Relevant documents retrieved to help answer the user query.
        :param is_use_tools: A boolean indicating whether to enable multi-agent system.
        :param uid: Unique identifier for the current user session, useful for logging and tracking.
        :return: Yields updated conversation history after each token generated, and the final response with sources.
        """
        logger.info(f"Preparing to generate a response based on context and history [uid - {uid}]")
        if not history or not history[-1].get("role"):
            yield history[:-1]
            return

        logger.info(f"Beginning response generation [uid - {uid}]")

        # Use log-based approach
        logger.info(f"Using log-based RAG approach [uid - {uid}]")
        response_text, _ = await self._log_based_approach(
            history, mode, retrieved_docs, uid, model, is_use_tools
        )

        logger.info(f"Response generation completed [uid - {uid}]")
        history.append({"role": "assistant", "content": response_text})
        yield history
        
        self.message_manager.queue -= 1

    async def _log_based_approach(self, history: List[dict], mode: str, retrieved_docs: str, uid: str, model: str, use_log_search: bool) -> tuple[str, list]:
        """Use MultiAgentManager for hybrid RAG+Logs approach"""
        
        # Update model if changed
        self.update_model(model)
        
        # Get the last user message
        last_user_message = history[-1].get("content", "") if history else ""
        
        # Use Multi-Agent System if logs are enabled and available
        if use_log_search and self.multi_agent_manager:
            try:
                logger.info(f"Using multi-agent system (RAG+Logs) with model {model} [uid - {uid}]")
                # Update model in multi-agent system
                self.multi_agent_manager.update_model(model)
                
                result = self.multi_agent_manager.process_query_sync(
                    query=last_user_message,
                    dialog_history=history,
                    thread_id=uid,
                    retrieved_docs=retrieved_docs
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



class UIManager:
    def __init__(self):
        self.message_manager: MessageManager = MessageManager()
        self.prompt_manager: SystemPromptManager = SystemPromptManager()
        self.analytics_manager: AnalyticsManager = AnalyticsManager()
        self.audio_manager: AudioManager = AudioManager()
        self.document_manager: DocumentManager = DocumentManager()
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
                        
                        with gr.Tab("CSV Логи из logs"):
                            with gr.Row():
                                user_request_time = gr.DateTime(
                                    label="Время запроса пользователя",
                                    value=None,
                                    info="Будут показаны логи от этого времени до 2 дней назад"
                                )
                            
                            with gr.Row():
                                with gr.Column():
                                    user_match_id = gr.Textbox(
                                        label="Match ID",
                                        placeholder="Например: 0, 1, 2...",
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
                        retrieved_docs = gr.HTML(
                            value="Появятся после задавания вопросов",
                            label="Извлеченные фрагменты",
                            show_label=True
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
                outputs=[modal, documents_tab, settings_tab, login_btn]
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

            # Load CSV logs from logs
            load_csv_logs_btn.click(
                fn=self.document_manager.load_csv_logs_from_data,
                inputs=[user_request_time, user_match_id],
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
                inputs=[chatbot, collection_radio, k_documents],
                outputs=[retrieved_docs],
                queue=True,
            ).success(
                fn=self.model_manager.generate_response_stream,
                inputs=[model, chatbot, collection_radio, retrieved_docs, is_use_tools, uid],
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
                    login_btn,
                    modal,
                    message_login,
                    files_selected
                ],
                js=LOCAL_STORAGE
            )

        demo.queue(max_size=128, api_open=False, default_concurrency_limit=5)
        return demo
