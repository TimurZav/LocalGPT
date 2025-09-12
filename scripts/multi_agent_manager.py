"""
Многоагентная система для обработки гибридных запросов RAG + Logs
Использует LangGraph для оркестрации агентов
"""
import logging
from enum import Enum
from dataclasses import dataclass
from claude_code_llm import ClaudeCodeLLM
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, List, Any, Optional, TypedDict, Annotated

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
    retrieved_docs: str  # Документы, полученные из UI
    query_type: str
    rag_result: Optional[AgentResult]
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
            ("system", """Ты RAG агент, работающий с документами. Отвечай ИСКЛЮЧИТЕЛЬНО на основе предоставленного контекста из документов.

            Инструкции:
            1. Используй ТОЛЬКО информацию из предоставленного контекста
            2. НЕ добавляй информацию из своих знаний, если её нет в контексте
            3. Если в контексте нет ответа на вопрос, честно скажи об этом
            4. Укажи источники для каждого факта из контекста
            5. Структурируй ответ логично и понятно
            6. ВАЖНО: Учитывай контекст предыдущих вопросов и ответов из истории диалога
            
            Цель: предоставить точный ответ, основанный исключительно на документах."""),
            ("human", "История диалога:\n{dialog_history}\n\nКонтекст из документов:\n{context}\n\nТекущий вопрос: {query}")
        ])


    def _create_agent_graph(self):
        """Создание графа агентов"""
        workflow = StateGraph(GraphState)
        
        # Добавление узлов
        workflow.add_node("classifier", self._classify_query)
        workflow.add_node("rag_agent", self._run_rag_agent)
        
        # Определение маршрутов - простая цепочка
        workflow.set_entry_point("classifier")
        workflow.add_edge("classifier", "rag_agent")
        workflow.add_edge("rag_agent", END)
        
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
            
            # Получение контекста из документов через переданные retrieved_docs
            rag_context = state.get("retrieved_docs", "")
            sources = ["documents"]  # Общий источник для переданных документов
            
            if not rag_context:
                rag_result = AgentResult(
                    agent_type="rag",
                    success=False,
                    content="Информация не найдена в документах",
                    sources=[],
                    metadata={"context_length": 0}
                )
            else:
                # Генерация ответа на основе контекста документов
                response = self.llm.invoke(
                    self.rag_agent_prompt.format_messages(
                        dialog_history=dialog_history,
                        context=rag_context,
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
            state["final_answer"] = rag_result.content
            state["sources"] = rag_result.sources
            logger.info(f"RAG агент завершен: success={rag_result.success}")
            return state
            
        except Exception as e:
            logger.error(f"Ошибка RAG агента: {e}")
            rag_result = AgentResult(
                agent_type="rag",
                success=False,
                content=f"Ошибка обработки документов: {str(e)}",
                sources=[],
                metadata={},
                error=str(e)
            )
            state["rag_result"] = rag_result
            state["final_answer"] = rag_result.content
            state["sources"] = rag_result.sources
            return state




    async def process_query(self, query: str, dialog_history: Optional[List[dict]] = None, thread_id: str = "default", retrieved_docs: str = "") -> Dict[str, Any]:
        """Основной метод обработки запроса"""
        try:
            # Инициализация состояния
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "original_query": query,
                "dialog_history": dialog_history,
                "retrieved_docs": retrieved_docs,
                "query_type": "",
                "rag_result": None,
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
                "metadata": final_state.get("metadata", {})
            }
            
        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {e}")
            return {
                "answer": f"Произошла ошибка при обработке запроса: {str(e)}",
                "sources": [],
                "query_type": "error",
                "rag_result": None,
                "metadata": {"error": str(e)}
            }

    def process_query_sync(self, query: str, dialog_history: Optional[List[dict]] = None, thread_id: str = "default", retrieved_docs: str = "") -> Dict[str, Any]:
        """Синхронная версия обработки запроса"""
        try:
            # Инициализация состояния
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "original_query": query,
                "dialog_history": dialog_history,
                "retrieved_docs": retrieved_docs,
                "query_type": "",
                "rag_result": None,
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
                "metadata": final_state.get("metadata", {})
            }
            
        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {e}")
            return {
                "answer": f"Произошла ошибка при обработке запроса: {str(e)}",
                "sources": [],
                "query_type": "error", 
                "rag_result": None,
                "metadata": {"error": str(e)}
            }