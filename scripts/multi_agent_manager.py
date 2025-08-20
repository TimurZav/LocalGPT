"""
Многоагентная система для обработки гибридных запросов RAG + Logs
Использует LangGraph для оркестрации агентов
"""

import logging
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from openrouter import ChatOpenRouter
from __init__ import MODELS

# Для LangGraph
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Типы запросов"""
    RAG_ONLY = "rag_only"
    LOGS_ONLY = "logs_only" 
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
    logs_result: Optional[AgentResult]
    final_answer: Optional[str]
    sources: List[str]
    metadata: Dict[str, Any]

class MultiAgentManager:
    """Менеджер многоагентной системы"""
    
    def __init__(self, document_manager, model_name: str = MODELS[0]):
        self.document_manager = document_manager
        self.model_name = model_name
        self.llm = ChatOpenRouter(model_name=model_name)
        
        # Создание графа агентов
        self.graph = self._create_agent_graph()
        
        # Промпты
        self.query_classifier_prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты классификатор запросов. Определи тип запроса:
 
            HYBRID - если вопрос требует данных И из документов, И из логов
            UNCLEAR - если неясно, какие источники нужны

            Примеры:
            - "Какие политики безопасности нарушались и какие инциденты происходили?" → HYBRID
            - "Привет" → UNCLEAR

            Отвечай только одним словом: HYBRID или UNCLEAR"""),
            ("human", "{query}")
        ])
        
        self.rag_agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты RAG агент. Создай КРАТКУЮ ВЫЖИМКУ из документов, которая дополнит анализ логов.

            Инструкции:
            1. Найди в чанках информацию, которая объясняет события из логов
            2. Ищи теоретическую информацию о механиках, правилах, ограничениях
            3. Включи информацию о наградах, героях, механиках игры
            4. ОСОБОЕ ВНИМАНИЕ: ищи информацию о сохранениях, синхронизации данных, потере прогресса
            5. Укажи источники для каждого факта
            6. Если есть результат анализа логов - найди документацию, которая это объясняет
            7. Формат: краткие пункты с объяснениями и контекстом
            
            Цель: дополнить практические данные из логов теоретическими знаниями."""),
            ("human", "Контекст из документов:\n{context}\n\nРезультат анализа логов:\n{logs_result}\n\nВопрос: {query}")
        ])
        
        self.logs_agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты агент анализа логов. Тебе предоставлены ВСЕ логи системы для анализа.
            
            Инструкции:
            1. Анализируй все предоставленные логи полностью
            2. Ищи паттерны, ошибки, события, связанные с вопросом пользователя
            3. Группируй похожие события вместе
            4. Указывай временные метки для важных событий
            5. Если в логах нет информации для ответа, так и скажи
            6. Предоставляй детальный анализ со статистикой, если возможно
            
            Будь максимально полезным и аналитическим."""),
            ("human", "Все логи системы:\n{logs}\n\nВопрос для анализа: {query}")
        ])
        
        self.integration_prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты интеграционный агент. Твоя задача - создать полный и связный ответ, объединив:
            1. Результат RAG поиска по документам (релевантные чанки из docx)
            2. Результат анализа всех логов системы

            Правила интеграции:
            1. Сначала используй информацию из документов для теоретической части ответа
            2. Затем дополни практической информацией из логов (события, ошибки, статистика)
            3. Создай связный нарратив, объединив оба источника
            4. Если результаты противоречат друг другу, укажи это и объясни различия
            5. Если один источник не содержит информации, полноценно используй другой
            6. Всегда указывай, откуда взята информация

            Структура ответа:
            [Основной интегрированный ответ с использованием обоих источников]
            
            ## Источники данных
            - Документы: [список файлов, если есть]
            - Системные логи: [краткая характеристика логов]"""),
            ("human", """Вопрос пользователя: {query}

            Информация из документов (RAG):
            {rag_result}

            Информация из всех логов системы:  
            {logs_result}

            Создай интегрированный ответ, объединив информацию из документов и логов.""")
        ])


    def _create_agent_graph(self):
        """Создание графа агентов"""
        workflow = StateGraph(GraphState)
        
        # Добавление узлов
        workflow.add_node("classifier", self._classify_query)
        workflow.add_node("rag_agent", self._run_rag_agent)
        workflow.add_node("logs_agent", self._run_logs_agent)
        workflow.add_node("integrator", self._integrate_results)
        
        # Определение маршрутов
        workflow.set_entry_point("classifier")
        
        workflow.add_conditional_edges(
            "classifier",
            self._route_query,
            {
                "rag_only": "rag_agent",
                "logs_only": "logs_agent",
                "hybrid": "logs_agent",  # Для гибридных сначала ЛОГИ
                "unclear": END
            }
        )
        
        workflow.add_conditional_edges(
            "rag_agent",
            self._after_rag,
            {
                "to_logs": "logs_agent",
                "to_integrator": "integrator",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "logs_agent",
            self._after_logs,
            {
                "to_rag": "rag_agent",
                "to_integrator": "integrator",
                "end": END
            }
        )
        workflow.add_edge("integrator", END)
        
        return workflow.compile(checkpointer=MemorySaver())

    def update_model(self, model_name: str):
        """Обновление модели для всех агентов"""
        if model_name != self.model_name:
            self.model_name = model_name
            self.llm = ChatOpenRouter(model_name=model_name)
            logger.info(f"Multi-agent system updated to use model: {model_name}")

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
        """Выполнение RAG агента для поиска по документам"""
        try:
            query = state["original_query"]
            
            # Получение контекста из документов через RAG поиск (напрямую)
            rag_context, sources = self.document_manager.get_rag_context(query, k_documents=8)
            
            if not rag_context:
                rag_result = AgentResult(
                    agent_type="rag",
                    success=False,
                    content="Информация не найдена в документах",
                    sources=[],
                    metadata={"context_length": 0}
                )
            else:
                # Получение результата логов, если есть
                logs_result = state.get("logs_result")
                logs_content = logs_result.content if logs_result and logs_result.success else "Анализ логов не выполнен"
                
                # Генерация ответа на основе контекста документов + результат логов
                response = self.llm.invoke(
                    self.rag_agent_prompt.format_messages(
                        context=rag_context,
                        logs_result=logs_content,
                        query=query
                    )
                )
                
                rag_result = AgentResult(
                    agent_type="rag",
                    success=True,
                    content=response.content,
                    sources=sources,
                    metadata={"context_length": len(rag_context), "sources_count": len(sources)}
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

    def _run_logs_agent(self, state: GraphState) -> GraphState:
        """Выполнение агента логов"""
        try:
            query = state["original_query"]
            
            # Получение всех логов
            if not self.document_manager.log_entries:
                logs_result = AgentResult(
                    agent_type="logs",
                    success=False,
                    content="Логи не загружены в систему",
                    sources=[],
                    metadata={}
                )
            else:
                # Генерация ответа на основе всех логов
                all_logs = "\n".join(self.document_manager.log_entries)
                
                response = self.llm.invoke(
                    self.logs_agent_prompt.format_messages(
                        logs=all_logs,
                        query=query
                    )
                )
                
                logs_result = AgentResult(
                    agent_type="logs",
                    success=True,
                    content=response.content,
                    sources=["logs"],
                    metadata={
                        "total_logs": len(self.document_manager.log_entries)
                    }
                )
            
            state["logs_result"] = logs_result
            logger.info(f"Агент логов завершен: success={logs_result.success}, processed {len(self.document_manager.log_entries) if self.document_manager.log_entries else 0} logs")
            return state
            
        except Exception as e:
            logger.error(f"Ошибка агента логов: {e}")
            state["logs_result"] = AgentResult(
                agent_type="logs",
                success=False,
                content=f"Ошибка анализа логов: {str(e)}",
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
            logs_result = state.get("logs_result")
            
            # Подготовка результатов для интеграции
            rag_content = rag_result.content if rag_result and rag_result.success else "Информация не найдена в документах"
            logs_content = logs_result.content if logs_result and logs_result.success else "Информация не найдена в логах"
            
            # Интеграция результатов
            response = self.llm.invoke(
                self.integration_prompt.format_messages(
                    query=query,
                    rag_result=rag_content,
                    logs_result=logs_content
                )
            )
            
            # Сбор всех источников
            all_sources = []
            if rag_result and rag_result.success:
                all_sources.extend(rag_result.sources)
            if logs_result and logs_result.success:
                all_sources.extend(logs_result.sources)
            
            state["final_answer"] = response.content
            state["sources"] = list(set(all_sources))  # Убираем дубликаты
            
            logger.info("Интеграция результатов завершена")
            return state
            
        except Exception as e:
            logger.error(f"Ошибка интеграции: {e}")
            # Возвращаем лучший доступный результат
            if state.get("rag_result") and state["rag_result"].success:
                state["final_answer"] = state["rag_result"].content
                state["sources"] = state["rag_result"].sources
            elif state.get("logs_result") and state["logs_result"].success:
                state["final_answer"] = state["logs_result"].content
                state["sources"] = state["logs_result"].sources
            else:
                state["final_answer"] = "Не удалось получить информацию из доступных источников"
                state["sources"] = []
            
            return state

    def _route_query(self, state: GraphState) -> str:
        """Маршрутизация после классификации"""
        query_type = state["query_type"]
        
        routing_map = {
            "rag_only": "rag_only",
            "logs_only": "logs_only", 
            "hybrid": "hybrid",
            "unclear": "unclear"
        }
        
        return routing_map.get(query_type, "unclear")

    def _after_rag(self, state: GraphState) -> str:
        """Определение следующего шага после RAG агента"""
        query_type = state["query_type"]
        
        if query_type == "hybrid":
            return "to_logs"  # Для гибридных запросов идем к агенту логов
        elif query_type == "rag_only":
            return "end"  # Для RAG-only завершаем
        else:
            return "to_integrator"  # Остальное в интегратор
    
    def _after_logs(self, state: GraphState) -> str:
        """Определение следующего шага после агента логов"""
        query_type = state["query_type"]
        
        if query_type == "hybrid":
            return "to_rag"  # Для гибридных запросов идем к RAG агенту с результатом логов
        elif query_type == "logs_only":
            return "end"  # Для logs-only завершаем
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
                "logs_result": None,
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
                "logs_result": final_state.get("logs_result"),
                "metadata": final_state.get("metadata", {})
            }
            
        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {e}")
            return {
                "answer": f"Произошла ошибка при обработке запроса: {str(e)}",
                "sources": [],
                "query_type": "error",
                "rag_result": None,
                "logs_result": None,
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
                "logs_result": None,
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
                "logs_result": final_state.get("logs_result"),
                "metadata": final_state.get("metadata", {})
            }
            
        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {e}")
            return {
                "answer": f"Произошла ошибка при обработке запроса: {str(e)}",
                "sources": [],
                "query_type": "error", 
                "rag_result": None,
                "logs_result": None,
                "metadata": {"error": str(e)}
            }