"""
Многоагентная система для обработки гибридных запросов RAG + Logs
Использует LangGraph для оркестрации агентов
"""
import logging
import tempfile
import os
from enum import Enum
from dataclasses import dataclass
from claude_code_llm import ClaudeCodeLLM
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Generator

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
    logs_result: Optional[AgentResult]
    final_answer: Optional[str]
    sources: List[str]
    metadata: Dict[str, Any]
    # Новые поля для streaming
    ui_history: Optional[List[dict]]  # История для UI с метаданными

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
            ("system", """Ты GraphRAG агент, работающий с Neo4j графовой базой данных. Создай КРАТКУЮ ВЫЖИМКУ из документов и связей между ними.

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
            ("human", "История диалога:\n{dialog_history}\n\nКонтекст из документов и графовых связей:\n{context}\n\nРезультат анализа логов:\n{logs_result}\n\nТекущий вопрос: {query}")
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
            1. Результат Neo4j GraphRAG поиска по документам (релевантные чанки и графовые связи)
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

            Информация из документов (GraphRAG):
            {rag_result}

            Информация из логов системы:  
            {logs_result}

            Создай интегрированный ответ, объединив всю информацию и учитывая контекст диалога.""")
        ])


    def _create_agent_graph(self):
        """Создание графа агентов"""
        workflow = StateGraph(GraphState)
        
        # Добавление узлов с индикаторами загрузки
        workflow.add_node("logs_loading", self._show_logs_loading)
        workflow.add_node("logs_agent", self._run_logs_agent)
        workflow.add_node("rag_loading", self._show_rag_loading)
        workflow.add_node("rag_agent", self._run_rag_agent)
        workflow.add_node("integration_loading", self._show_integration_loading)
        workflow.add_node("integrator", self._integrate_results)
        
        # Определение маршрутов - цепочка с индикаторами
        workflow.set_entry_point("logs_loading")
        workflow.add_edge("logs_loading", "logs_agent")
        workflow.add_edge("logs_agent", "rag_loading")
        workflow.add_edge("rag_loading", "rag_agent")
        workflow.add_edge("rag_agent", "integration_loading")
        workflow.add_edge("integration_loading", "integrator")
        workflow.add_edge("integrator", END)
        
        return workflow.compile(checkpointer=MemorySaver())

    def update_model(self, model_name: str):
        """Обновление модели для всех агентов"""
        if model_name != self.model_name:
            self.model_name = model_name
            self.llm = ClaudeCodeLLM(model_name=model_name)
            logger.info(f"Multi-agent system updated to use model: {model_name}")

    @staticmethod
    def _format_dialog_history(dialog_history: Optional[List[dict]]) -> str:
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

    @staticmethod
    def _show_logs_loading(state: GraphState) -> GraphState:
        """Показать индикатор загрузки для анализа логов"""
        if state.get("ui_history") is not None:
            ui_history = state["ui_history"].copy()
            ui_history.append({
                "role": "assistant",
                "content": "🔍 Анализирую логи системы...",
                "metadata": {"title": "⏳ Анализ логов"}
            })
            state["ui_history"] = ui_history
        return state

    @staticmethod
    def _show_rag_loading(state: GraphState) -> GraphState:
        """Показать индикатор загрузки для поиска в документах"""
        if state.get("ui_history") is not None:
            ui_history = state["ui_history"].copy()
            ui_history.append({
                "role": "assistant",
                "content": "📚 Ищу информацию в документах...",
                "metadata": {"title": "⏳ Поиск в документах"}
            })
            state["ui_history"] = ui_history
        return state

    @staticmethod
    def _show_integration_loading(state: GraphState) -> GraphState:
        """Показать индикатор загрузки для интеграции результатов"""
        if state.get("ui_history") is not None:
            ui_history = state["ui_history"].copy()
            ui_history.append({
                "role": "assistant",
                "content": "🔗 Интегрирую результаты и создаю финальный ответ...",
                "metadata": {"title": "⏳ Интеграция результатов"}
            })
            state["ui_history"] = ui_history
        return state

    def _run_rag_agent(self, state: GraphState) -> GraphState:
        """Выполнение RAG агента для поиска по документах"""
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
            
            # Заменяем индикатор загрузки на результат
            if state.get("ui_history") is not None and rag_result.success:
                ui_history = state["ui_history"].copy()
                # Заменяем индикатор загрузки на результат
                if ui_history and ui_history[-1].get("metadata", {}).get("title") == "⏳ Поиск в документах":
                    ui_history[-1] = {
                        "role": "assistant",
                        "content": rag_result.content[:200] + "..." if len(rag_result.content) > 200 else rag_result.content,
                        "metadata": {"title": "📚 Результат поиска в документах"}
                    }
                state["ui_history"] = ui_history
            
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
                # # Ограничение логов для избежания ошибки "Argument list too long"
                # Записываем ВСЕ логи в постоянный файл
                log_entries = self.document_manager.log_entries
                all_logs = "\n".join(log_entries)
                
                # Создаем временный файл для логов чтобы обойти ограничение длины аргументов
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as temp_file:
                    temp_file.write(all_logs)
                    temp_file_path = temp_file.name
                    temp_file_path = temp_file.name

                logger.info(f"Обрабатываем {len(log_entries)} записей логов, размер: {len(all_logs)} символов")
                logger.info(f"Логи сохранены во временный файл: {temp_file_path}")
                # Теперь передаем путь к файлу вместо содержимого
                
                try:
                    response = self.llm.invoke(
                        self.logs_agent_prompt.format_messages(
                            dialog_history=dialog_history,
                            logs=f"ФАЙЛ С ЛОГАМИ: {temp_file_path}\n\nИспользуйте инструмент Read для чтения файла: Read {temp_file_path}",
                            query=query
                        )
                    )
                finally:
                    # Удаляем временный файл после использования
                    try:
                        os.unlink(temp_file_path)
                        logger.info(f"Временный файл удален: {temp_file_path}")
                    except Exception as e:
                        logger.warning(f"Не удалось удалить временный файл {temp_file_path}: {e}")
            
                
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
            
            # Заменяем индикатор загрузки на результат
            if state.get("ui_history") is not None and logs_result.success:
                ui_history = state["ui_history"].copy()
                # Заменяем индикатор загрузки на результат
                if ui_history and ui_history[-1].get("metadata", {}).get("title") == "⏳ Анализ логов":
                    ui_history[-1] = {
                        "role": "assistant",
                        "content": logs_result.content[:200] + "..." if len(logs_result.content) > 200 else logs_result.content,
                        "metadata": {"title": "📊 Результат анализа логов"}
                    }
                state["ui_history"] = ui_history
            
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
            
            final_answer = response.content + "\n\n*Использованы данные из документов и логов*"
            state["final_answer"] = final_answer
            state["sources"] = list(set(all_sources))  # Убираем дубликаты
            
            # Заменяем индикатор загрузки на финальный ответ
            if state.get("ui_history") is not None:
                ui_history = state["ui_history"].copy()
                # Заменяем индикатор загрузки на финальный ответ
                if ui_history and ui_history[-1].get("metadata", {}).get("title") == "⏳ Интеграция результатов":
                    ui_history[-1] = {"role": "assistant", "content": final_answer}
                state["ui_history"] = ui_history
            
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

    def process_query_with_streaming(
        self, 
        query: str, 
        dialog_history: Optional[List[dict]] = None,
        thread_id: str = "default",
        retrieved_docs: str = ""
    ) -> Generator[List[dict], None, None]:
        """
        Streaming версия с использованием LangGraph workflow
        """
        try:
            # Инициализация состояния для workflow  
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "original_query": query,
                "dialog_history": dialog_history,
                "retrieved_docs": retrieved_docs,
                "query_type": "",
                "rag_result": None,
                "logs_result": None,
                "final_answer": None,
                "sources": [],
                "metadata": {},
                "ui_history": dialog_history.copy() if dialog_history else []
            }
            
            # Выполнение workflow с потоковой передачей
            config = {"configurable": {"thread_id": thread_id}}
            
            # Используем stream вместо invoke для получения промежуточных результатов
            for chunk in self.graph.stream(initial_state, config):
                # chunk содержит результаты каждого шага
                node_name = list(chunk.keys())[0]
                node_state = chunk[node_name]
                
                # Получаем обновленную историю UI из состояния узла
                if "ui_history" in node_state:
                    yield node_state["ui_history"]
            
        except Exception as e:
            logger.error(f"Ошибка в workflow streaming: {e}")
            error_history = (dialog_history.copy() if dialog_history else [])
            error_history.append({
                "role": "assistant",
                "content": f"Произошла ошибка: {str(e)}",
                "metadata": {"title": "❌ Ошибка"}
            })
            yield error_history