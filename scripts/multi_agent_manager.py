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
from claude_code_llm import ClaudeCodeLLM
# from openrouter import ChatOpenRouter
# from __init__ import MODELS

# Для LangGraph
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Типы запросов"""
    HYBRID = "hybrid"

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
    dialog_history: Optional[List[dict]]  # История диалога для контекста
    query_type: str
    rag_result: Optional[AgentResult]
    logs_result: Optional[AgentResult]
    final_answer: Optional[str]
    sources: List[str]
    metadata: Dict[str, Any]

class MultiAgentManager:
    """Менеджер многоагентной системы"""
    
    def __init__(self, document_manager, model_name: str = "claude-3-5-sonnet-20241022"):
        self.document_manager = document_manager
        self.model_name = model_name
        self.llm = ClaudeCodeLLM(model_name=model_name)
        
        # Создание графа агентов
        self.graph = self._create_agent_graph()
        
        # Промпты (классификатор больше не нужен, всегда используем HYBRID)
        
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
            8. ВАЖНО: Учитывай контекст предыдущих вопросов и ответов из истории диалога
            
            Цель: дополнить практические данные из логов теоретическими знаниями."""),
            ("human", "История диалога:\n{dialog_history}\n\nКонтекст из документов:\n{context}\n\nРезультат анализа логов:\n{logs_result}\n\nТекущий вопрос: {query}")
        ])
        
        self.logs_agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты агент анализа логов. Тебе предоставлены логи системы для анализа.
            
            Инструкции:
            1. Анализируй все предоставленные логи полностью
            2. Ищи паттерны, ошибки, события, связанные с вопросом пользователя
            3. Группируй похожие события вместе
            4. Указывай временные метки для важных событий
            5. Если в логах нет информации для ответа, так и скажи
            6. Предоставляй детальный анализ со статистикой, если возможно
            7. ВАЖНО: Учитывай контекст предыдущих вопросов и ответов из истории диалога
            
            Будь максимально полезным и аналитическим."""),
            ("human", "История диалога:\n{dialog_history}\n\nЛоги системы:\n{logs}\n\nТекущий вопрос для анализа: {query}")
        ])
        
        self.integration_prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты интеграционный агент. Твоя задача - создать полный и связный ответ, объединив:
            1. Результат RAG поиска по документам (релевантные чанки из docx)
            2. Результат анализа всех логов системы
            3. Контекст предыдущего диалога для непрерывности разговора

            Правила интеграции:
            1. Сначала используй информацию из документов для теоретической части ответа
            2. Затем дополни практической информацией из логов (события, ошибки, статистика)
            3. Создай связный нарратив, объединив оба источника
            4. Если результаты противоречат друг другу, укажи это и объясни различия
            5. Если один источник не содержит информации, полноценно используй другой
            6. Всегда указывай, откуда взята информация
            7. ВАЖНО: Учитывай контекст предыдущих вопросов и ответов для логичности диалога
            8. Ссылайся на предыдущие ответы, если текущий вопрос связан с ними

            Структура ответа:
            [Основной интегрированный ответ с использованием обоих источников и контекста диалога]
            
            ## Источники данных
            - Документы: [список файлов, если есть]
            - Системные логи: [краткая характеристика логов]"""),
            ("human", """История диалога:
            {dialog_history}

            Текущий вопрос пользователя: {query}

            Информация из документов (RAG):
            {rag_result}

            Информация из логов системы:  
            {logs_result}

            Создай интегрированный ответ, объединив всю информацию и учитывая контекст диалога.""")
        ])


    def _create_agent_graph(self):
        """Создание графа агентов"""
        workflow = StateGraph(GraphState)
        
        # Добавление узлов
        workflow.add_node("classifier", self._classify_query)
        workflow.add_node("rag_agent", self._run_rag_agent)
        workflow.add_node("logs_agent", self._run_logs_agent)
        workflow.add_node("integrator", self._integrate_results)
        
        # Определение маршрутов - простая цепочка
        workflow.set_entry_point("classifier")
        workflow.add_edge("classifier", "logs_agent")
        workflow.add_edge("logs_agent", "rag_agent")
        workflow.add_edge("rag_agent", "integrator")
        workflow.add_edge("integrator", END)
        
        return workflow.compile(checkpointer=MemorySaver())

    def update_model(self, model_name: str):
        """Обновление модели для всех агентов"""
        if model_name != self.model_name:
            self.model_name = model_name
            self.llm = ClaudeCodeLLM(model_name=model_name)
            logger.info(f"Multi-agent system updated to use model: {model_name}")

    def _format_dialog_history(self, dialog_history: Optional[List[dict]]) -> str:
        """Форматирование истории диалога для промптов"""
        if not dialog_history or len(dialog_history) < 2:
            return "Нет предыдущих сообщений в диалоге."
        
        # Берем последние несколько пар вопрос-ответ (максимум 3 пары)
        max_pairs = 3
        formatted_history = []
        
        # Исключаем последнее сообщение пользователя (это текущий вопрос)
        history_to_process = dialog_history[:-1]
        
        # Группируем по парам user -> assistant
        pairs = []
        current_pair = {}
        
        for message in history_to_process:
            if message["role"] == "user":
                if current_pair.get("assistant"):
                    pairs.append(current_pair)
                current_pair = {"user": message["content"]}
            elif message["role"] == "assistant" and current_pair.get("user"):
                current_pair["assistant"] = message["content"]
        
        if current_pair.get("user") and current_pair.get("assistant"):
            pairs.append(current_pair)
        
        # Берем последние пары и форматируем
        recent_pairs = pairs[-max_pairs:] if len(pairs) > max_pairs else pairs
        
        for i, pair in enumerate(recent_pairs, 1):
            formatted_history.append(f"Вопрос {i}: {pair['user']}")
            formatted_history.append(f"Ответ {i}: {pair['assistant'][:200]}...")  # Обрезаем длинные ответы
        
        return "\n\n".join(formatted_history) if formatted_history else "Нет предыдущих сообщений в диалоге."

    def _classify_query(self, state: GraphState) -> GraphState:
        """Классификация запроса - всегда возвращает HYBRID"""
        state["query_type"] = QueryType.HYBRID.value
        state["metadata"] = {"classification_confidence": "high"}
        logger.info("Запрос классифицирован как HYBRID")
        return state

    def _run_rag_agent(self, state: GraphState) -> GraphState:
        """Выполнение RAG агента для поиска по документам"""
        try:
            query = state["original_query"]
            dialog_history = self._format_dialog_history(state.get("dialog_history"))
            
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
                        dialog_history=dialog_history,
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
            dialog_history = self._format_dialog_history(state.get("dialog_history"))
            
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
                
                logger.info(f"Обрабатываем {len(log_entries)} записей логов, размер: {len(all_logs)} символов")
                
                response = self.llm.invoke(
                    self.logs_agent_prompt.format_messages(
                        dialog_history=dialog_history,
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
            dialog_history = self._format_dialog_history(state.get("dialog_history"))
            rag_result = state.get("rag_result")
            logs_result = state.get("logs_result")
            
            # Подготовка результатов для интеграции
            rag_content = rag_result.content if rag_result and rag_result.success else "Информация не найдена в документах"
            logs_content = logs_result.content if logs_result and logs_result.success else "Информация не найдена в логах"
            
            # Интеграция результатов
            response = self.llm.invoke(
                self.integration_prompt.format_messages(
                    dialog_history=dialog_history,
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



    async def process_query(self, query: str, dialog_history: Optional[List[dict]] = None, thread_id: str = "default") -> Dict[str, Any]:
        """Основной метод обработки запроса"""
        try:
            # Инициализация состояния
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "original_query": query,
                "dialog_history": dialog_history,
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

    def process_query_sync(self, query: str, dialog_history: Optional[List[dict]] = None, thread_id: str = "default") -> Dict[str, Any]:
        """Синхронная версия обработки запроса"""
        try:
            # Инициализация состояния
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "original_query": query,
                "dialog_history": dialog_history,
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