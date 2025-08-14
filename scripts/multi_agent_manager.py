"""
Многоагентная система для обработки гибридных запросов RAG + SQL
Использует LangGraph для оркестрации агентов
"""

import json
import logging
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from openrouter import ChatOpenRouter

# Для LangGraph
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Типы запросов"""
    RAG_ONLY = "rag_only"
    SQL_ONLY = "sql_only" 
    HYBRID = "hybrid"
    UNCLEAR = "unclear"

@dataclass
class AgentResult:
    """Результат работы агента"""
    agent_type: str
    success: bool
    content: str
    sources: List[str]
    metadata: Dict[str, Any]
    error: Optional[str] = None

class GraphState(TypedDict):
    """Состояние графа агентов"""
    messages: Annotated[List, add_messages]
    original_query: str
    query_type: str
    rag_result: Optional[AgentResult]
    sql_result: Optional[AgentResult]
    final_answer: Optional[str]
    sources: List[str]
    metadata: Dict[str, Any]

class MultiAgentManager:
    """Менеджер многоагентной системы"""
    
    def __init__(self, document_manager, database_url: str, model_name: str = "openai/gpt-4o-mini"):
        self.document_manager = document_manager
        self.model_name = model_name
        self.llm = ChatOpenRouter(model_name=model_name)
        
        # Инициализация SQL агента
        self.sql_db = SQLDatabase.from_uri(database_url, sample_rows_in_table_info=3)
        self.sql_agent = self._create_sql_agent()
        
        # Создание графа агентов
        self.graph = self._create_agent_graph()
        
        # Промпты
        self.query_classifier_prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты классификатор запросов. Определи тип запроса:

            RAG_ONLY - если вопрос касается только информации из документов (политики, планы, процедуры, техническая документация)
            SQL_ONLY - если вопрос касается только данных из базы (пользователи, проекты, задачи, оборудование, инциденты)  
            HYBRID - если вопрос требует данных И из документов, И из базы данных
            UNCLEAR - если неясно, какие источники нужны

            Примеры:
            - "Какие требования к паролям?" → RAG_ONLY
            - "Кто работает в отделе разработки?" → SQL_ONLY  
            - "Какие проекты по безопасности ведутся и какие политики действуют?" → HYBRID
            - "Привет" → UNCLEAR

            Отвечай только одним словом: RAG_ONLY, SQL_ONLY, HYBRID или UNCLEAR"""),
            ("human", "{query}")
        ])
        
        self.rag_agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты RAG агент. Используй ТОЛЬКО предоставленный контекст из документов для ответа на вопрос.
            
            Если контекст не содержит информации для ответа, скажи "Информация не найдена в документах".
            Всегда указывай источники информации.
            Будь точным и конкретным."""),
            ("human", "Контекст из документов:\n{context}\n\nВопрос: {query}")
        ])
        
        self.integration_prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты интеграционный агент. Объедини результаты RAG и SQL агентов в единый ответ.

            Правила:
            1. Используй ВСЮ информацию из обоих источников
            2. Структурируй ответ логично 
            3. Укажи источники данных
            4. Если результаты противоречат друг другу, укажи это
            5. Если один из агентов не нашел информации, используй результат другого

            Формат ответа:
            ## Ответ
            [Основная информация]

            ## Источники
            - Документы: [список файлов]
            - База данных: [типы данных]"""),
            ("human", """Вопрос: {query}

            Результат RAG агента:
            {rag_result}

            Результат SQL агента:  
            {sql_result}

            Объедини эти результаты в полный ответ на вопрос.""")
        ])

    def _create_sql_agent(self):
        """Создание SQL агента"""
        system_prompt = """You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct PostgreSQL query to run, then look at the results of the query and return the answer.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for the relevant columns given the question.
        You have access to tools for interacting with the database.
        Only use the given tools. Only use the information returned by the tools to construct your final answer.
        You MUST double check your query before executing it. If you get an error while executing a query then you should stop!

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
        Only create an SQL statement ONCE!
        
        Always provide specific data from the database, not just general statements."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        return create_sql_agent(
            self.llm,
            db=self.sql_db,
            prompt=prompt,
            agent_type="tool-calling",
            verbose=True
        )

    def _create_agent_graph(self):
        """Создание графа агентов"""
        workflow = StateGraph(GraphState)
        
        # Добавление узлов
        workflow.add_node("classifier", self._classify_query)
        workflow.add_node("rag_agent", self._run_rag_agent)
        workflow.add_node("sql_agent", self._run_sql_agent)
        workflow.add_node("integrator", self._integrate_results)
        
        # Определение маршрутов
        workflow.set_entry_point("classifier")
        
        workflow.add_conditional_edges(
            "classifier",
            self._route_query,
            {
                "rag_only": "rag_agent",
                "sql_only": "sql_agent",
                "hybrid": "rag_agent",  # Для гибридных сначала RAG
                "unclear": END
            }
        )
        
        workflow.add_conditional_edges(
            "rag_agent",
            self._after_rag,
            {
                "to_sql": "sql_agent",
                "to_integrator": "integrator",
                "end": END
            }
        )
        
        workflow.add_edge("sql_agent", "integrator")
        workflow.add_edge("integrator", END)
        
        return workflow.compile(checkpointer=MemorySaver())

    def _classify_query(self, state: GraphState) -> GraphState:
        """Классификация запроса"""
        try:
            query = state["original_query"]
            response = self.llm.invoke(self.query_classifier_prompt.format_messages(query=query))
            query_type = response.content.strip().upper()
            
            if query_type not in [qt.value.upper() for qt in QueryType]:
                query_type = QueryType.UNCLEAR.value.upper()
            
            state["query_type"] = query_type.lower()
            state["metadata"] = {"classification_confidence": "high"}
            
            logger.info(f"Классификация запроса: {query_type}")
            return state
            
        except Exception as e:
            logger.error(f"Ошибка классификации: {e}")
            state["query_type"] = QueryType.UNCLEAR.value
            return state

    def _run_rag_agent(self, state: GraphState) -> GraphState:
        """Выполнение RAG агента"""
        try:
            query = state["original_query"]
            
            # Получение контекста из документов
            retrieved_docs, scores = self.document_manager.retrieve_documents(
                history=[{"role": "user", "content": query}],
                collection_radio="RAG",  # Предполагаем RAG режим
                k_documents=6,
                uid="multi_agent"
            )
            
            if not retrieved_docs or retrieved_docs == "Появятся после задавания вопросов":
                rag_result = AgentResult(
                    agent_type="rag",
                    success=False,
                    content="Информация не найдена в документах",
                    sources=[],
                    metadata={"scores": []}
                )
            else:
                # Генерация ответа на основе контекста
                response = self.llm.invoke(
                    self.rag_agent_prompt.format_messages(
                        context=retrieved_docs,
                        query=query
                    )
                )
                
                # Извлечение источников из retrieved_docs
                import re
                sources = re.findall(r'<a\s+[^>]*>(.*?)</a>', retrieved_docs)
                
                rag_result = AgentResult(
                    agent_type="rag",
                    success=True,
                    content=response.content,
                    sources=sources,
                    metadata={"scores": scores, "context_length": len(retrieved_docs)}
                )
            
            state["rag_result"] = rag_result
            logger.info(f"RAG агент завершен: success={rag_result.success}")
            return state
            
        except Exception as e:
            logger.error(f"Ошибка RAG агента: {e}")
            state["rag_result"] = AgentResult(
                agent_type="rag",
                success=False,
                content=f"Ошибка обработки документов: {str(e)}",
                sources=[],
                metadata={},
                error=str(e)
            )
            return state

    def _run_sql_agent(self, state: GraphState) -> GraphState:
        """Выполнение SQL агента"""
        try:
            query = state["original_query"]
            
            # Выполнение SQL агента
            result = self.sql_agent.invoke({"input": query})
            
            # Очистка результата от <think> тегов
            import re
            clean_output = re.sub(r'<think>.*?</think>', '', result["output"], flags=re.DOTALL).strip()
            
            sql_result = AgentResult(
                agent_type="sql",
                success=True,
                content=clean_output,
                sources=["database"],
                metadata={"raw_output": result["output"]}
            )
            
            state["sql_result"] = sql_result
            logger.info("SQL агент завершен успешно")
            return state
            
        except Exception as e:
            logger.error(f"Ошибка SQL агента: {e}")
            state["sql_result"] = AgentResult(
                agent_type="sql",
                success=False,
                content=f"Ошибка выполнения запроса к базе данных: {str(e)}",
                sources=[],
                metadata={},
                error=str(e)
            )
            return state

    def _integrate_results(self, state: GraphState) -> GraphState:
        """Интеграция результатов"""
        try:
            query = state["original_query"]
            rag_result = state.get("rag_result")
            sql_result = state.get("sql_result")
            
            # Подготовка результатов для интеграции
            rag_content = rag_result.content if rag_result and rag_result.success else "Информация не найдена в документах"
            sql_content = sql_result.content if sql_result and sql_result.success else "Информация не найдена в базе данных"
            
            # Интеграция результатов
            response = self.llm.invoke(
                self.integration_prompt.format_messages(
                    query=query,
                    rag_result=rag_content,
                    sql_result=sql_content
                )
            )
            
            # Сбор всех источников
            all_sources = []
            if rag_result and rag_result.success:
                all_sources.extend(rag_result.sources)
            if sql_result and sql_result.success:
                all_sources.extend(sql_result.sources)
            
            state["final_answer"] = response.content
            state["sources"] = list(set(all_sources))  # Убираем дубликаты
            
            logger.info("Интеграция результатов завершена")
            return state
            
        except Exception as e:
            logger.error(f"Ошибка интеграции: {e}")
            # Возвращаем лучший доступный результат
            if state.get("rag_result") and state["rag_result"].success:
                state["final_answer"] = state["rag_result"].content
            elif state.get("sql_result") and state["sql_result"].success:
                state["final_answer"] = state["sql_result"].content
            else:
                state["final_answer"] = "Не удалось получить информацию из доступных источников"
            
            return state

    def _route_query(self, state: GraphState) -> str:
        """Маршрутизация после классификации"""
        query_type = state["query_type"]
        
        routing_map = {
            "rag_only": "rag_only",
            "sql_only": "sql_only", 
            "hybrid": "hybrid",
            "unclear": "unclear"
        }
        
        return routing_map.get(query_type, "unclear")

    def _after_rag(self, state: GraphState) -> str:
        """Определение следующего шага после RAG агента"""
        query_type = state["query_type"]
        
        if query_type == "hybrid":
            return "to_sql"  # Для гибридных запросов идем к SQL агенту
        elif query_type == "rag_only":
            return "end"  # Для RAG-only завершаем
        else:
            return "to_integrator"  # Остальное в интегратор

    async def process_query(self, query: str, thread_id: str = "default") -> Dict[str, Any]:
        """Основной метод обработки запроса"""
        try:
            # Инициализация состояния
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "original_query": query,
                "query_type": "",
                "rag_result": None,
                "sql_result": None,
                "final_answer": None,
                "sources": [],
                "metadata": {}
            }
            
            # Выполнение графа
            config = {"configurable": {"thread_id": thread_id}}
            final_state = await self.graph.ainvoke(initial_state, config)
            
            # Формирование результата
            return {
                "answer": final_state.get("final_answer", "Ответ не получен"),
                "sources": final_state.get("sources", []),
                "query_type": final_state.get("query_type", "unknown"),
                "rag_result": final_state.get("rag_result"),
                "sql_result": final_state.get("sql_result"),
                "metadata": final_state.get("metadata", {})
            }
            
        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {e}")
            return {
                "answer": f"Произошла ошибка при обработке запроса: {str(e)}",
                "sources": [],
                "query_type": "error",
                "rag_result": None,
                "sql_result": None,
                "metadata": {"error": str(e)}
            }

    def process_query_sync(self, query: str, thread_id: str = "default") -> Dict[str, Any]:
        """Синхронная версия обработки запроса"""
        try:
            # Инициализация состояния
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "original_query": query,
                "query_type": "",
                "rag_result": None,
                "sql_result": None,
                "final_answer": None,
                "sources": [],
                "metadata": {}
            }
            
            # Выполнение графа синхронно
            config = {"configurable": {"thread_id": thread_id}}
            final_state = self.graph.invoke(initial_state, config)
            
            # Формирование результата
            return {
                "answer": final_state.get("final_answer", "Ответ не получен"),
                "sources": final_state.get("sources", []),
                "query_type": final_state.get("query_type", "unknown"),
                "rag_result": final_state.get("rag_result"),
                "sql_result": final_state.get("sql_result"),
                "metadata": final_state.get("metadata", {})
            }
            
        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {e}")
            return {
                "answer": f"Произошла ошибка при обработке запроса: {str(e)}",
                "sources": [],
                "query_type": "error", 
                "rag_result": None,
                "sql_result": None,
                "metadata": {"error": str(e)}
            }