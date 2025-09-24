"""
Многоагентная система для обработки гибридных запросов RAG + Logs
Использует Semantic Kernel ChatCompletionAgent для оркестрации агентов
"""
import os
import logging
import asyncio
from dataclasses import dataclass
from claude_code_llm import ClaudeCodeLLM
from typing import Dict, List, Any, Optional, Generator
from langsmith_config import trace_function, is_tracing_enabled

# Semantic Kernel imports
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Результат работы агента"""
    agent_type: str
    success: bool
    content: str
    sources: List[str]
    metadata: Dict[str, Any]
    error: Optional[str] = None


class MultiAgentManager:
    """Менеджер многоагентной системы на основе Semantic Kernel ChatCompletionAgent"""
    
    def __init__(self, document_manager, model_name: str = "claude-3-5-sonnet-20241022"):
        self.document_manager = document_manager
        self.model_name = model_name
        self.llm = ClaudeCodeLLM(model_name=model_name)
        
        # Агенты будут созданы при первом запросе
        self.logs_agent = self._create_logs_agent()
        self.rag_agent = self._create_rag_agent()
        self.integration_agent = self._create_integration_agent()
        self.thread = self._create_thread()  # ChatHistoryAgentThread для управления диалогом
        
        # Текущее состояние обработки
        self.current_query = None
        self.current_dialog_history = None
        self.current_retrieved_docs = None
        self.ui_history = []
    
    @staticmethod
    def _create_chat_completion_service():
        """Создание chat completion service для агентов"""
        try:
            # Пробуем добавить сервис, если есть ключи
            api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                service = OpenAIChatCompletion(
                    ai_model_id="gpt-4o-mini",  # fallback модель
                    api_key=api_key,
                )
                logger.info("✅ Chat completion service настроен с OpenAI API")
                return service
        except Exception as e:
            logger.warning(f"⚠️ Не удалось настроить API: {e}")
            raise Exception("No valid API service available")
    
    def _create_logs_agent(self) -> ChatCompletionAgent:
        """Создание агента анализа логов"""
        return ChatCompletionAgent(
            service=self._create_chat_completion_service(),
            name="LogsAnalyst",
            instructions="""Ты агент анализа логов. Анализируй предоставленные логи системы.
            
            Задачи:
            1. Найди паттерны, ошибки, события, связанные с вопросом пользователя
            2. Группируй похожие события
            3. Указывай временные метки для важных событий
            4. Предоставляй детальную статистику
            5. Учитывай контекст предыдущих сообщений
            
            Если в логах нет информации для ответа, четко об этом сообщи.
            Будь аналитическим и полезным."""
        )
    
    def _create_rag_agent(self) -> ChatCompletionAgent:
        """Создание RAG агента для работы с документами"""
        return ChatCompletionAgent(
            service=self._create_chat_completion_service(),
            name="DocumentAnalyst", 
            instructions="""Ты GraphRAG агент для анализа документов и графовых связей.
            
            Задачи:
            1. Анализируй документы и связи между ними
            2. Ищи теоретическую информацию о механиках, правилах, ограничениях
            3. Находи информацию о сохранениях, синхронизации данных, потере прогресса
            4. Связывай документацию с событиями из логов
            5. Указывай источники для каждого факта
            6. Учитывай контекст диалога
            
            Формат: краткие пункты с объяснениями.
            Цель: дополнить данные из логов теоретическими знаниями."""
        )
    
    def _create_integration_agent(self) -> ChatCompletionAgent:
        """Создание интеграционного агента"""
        return ChatCompletionAgent(
            service=self._create_chat_completion_service(),
            name="IntegrationSpecialist",
            instructions="""Ты интеграционный агент. Создавай полные ответы, объединяя:
            1. Результаты анализа логов
            2. Информацию из документов
            3. Контекст предыдущего диалога

            Правила:
            1. Используй документы для теоретической части
            2. Дополняй практической информацией из логов
            3. Создавай связный нарратив
            4. При противоречиях объясняй различия
            5. Указывай источники информации
            6. Ссылайся на предыдущие ответы при необходимости

            Структура ответа:
            [Основной интегрированный ответ]
            
            ## Источники данных
            - Документы: [если есть]
            - Системные логи: [характеристика]"""
        )
    
    @staticmethod
    def _create_thread() -> ChatHistoryAgentThread:
        """Создание thread для управления диалогом между агентами"""
        return ChatHistoryAgentThread()
    
    @staticmethod
    def _format_dialog_history(dialog_history) -> str:
        """Форматирование истории диалога"""
        if not dialog_history or len(dialog_history) < 2:
            return "Нет предыдущих сообщений в диалоге."
        
        # Берем последние 3 пары вопрос-ответ
        max_pairs = 3
        formatted_history = []
        
        # Исключаем последнее сообщение пользователя (текущий вопрос)
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
        
        # Форматируем последние пары
        recent_pairs = pairs[-max_pairs:] if len(pairs) > max_pairs else pairs
        
        for i, pair in enumerate(recent_pairs, 1):
            formatted_history.append(f"Вопрос {i}: {pair['user']}")
            formatted_history.append(f"Ответ {i}: {pair['assistant'][:200]}...")
        
        return "\n\n".join(formatted_history) if formatted_history else "Нет предыдущих сообщений в диалоге."
    
    def _prepare_context_message(self, query, dialog_history, retrieved_docs) -> str:
        """Подготовка контекстного сообщения для агентов"""
        formatted_history = self._format_dialog_history(dialog_history)
        
        # Подготовка логов
        if self.document_manager.log_entries:
            logs_count = len(self.document_manager.log_entries)
            logs_sample = "\n".join(self.document_manager.log_entries[:5])  # Первые 5 записей как пример
            logs_info = f"Доступно {logs_count} записей логов. Пример:\n{logs_sample}\n[...остальные {logs_count-5} записей]"
        else:
            logs_info = "Логи не загружены в систему"
        
        # Формирование полного контекста
        context_message = f"""
КОНТЕКСТ ЗАПРОСА:

История диалога:
{formatted_history}

Системные логи:
{logs_info}

Документы и графовые связи:
{retrieved_docs if retrieved_docs else "Документы не предоставлены"}

ТЕКУЩИЙ ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{query}

Проанализируйте информацию согласно вашей роли и предоставьте результат.
"""
        return context_message
    
    def _prepare_logs_context(self, query: str, dialog_history) -> str:
        """Подготовка контекста для LogsAnalyst - только логи и история диалога"""
        formatted_history = self._format_dialog_history(dialog_history)
        
        # Подготовка логов
        if self.document_manager.log_entries:
            logs_count = len(self.document_manager.log_entries)
            # Для агента логов даем полную информацию о логах
            if logs_count <= 10:
                logs_info = "\n".join(self.document_manager.log_entries)
            else:
                # Если логов много, берем первые 10 и последние 5
                first_logs = "\n".join(self.document_manager.log_entries[:10])
                last_logs = "\n".join(self.document_manager.log_entries[-5:])
                logs_info = f"{first_logs}\n\n[...пропущено {logs_count-15} записей...]\n\n{last_logs}"
        else:
            logs_info = "Логи не загружены в систему"
        
        context_message = f"""
АНАЛИЗ ЛОГОВ СИСТЕМЫ:

История диалога:
{formatted_history}

СИСТЕМНЫЕ ЛОГИ:
{logs_info}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{query}

Проанализируй логи и найди информацию, связанную с вопросом пользователя. Сосредоточься на:
- Паттернах и ошибках в логах
- Временных метках важных событий
- Статистике и группировке событий
- Связи событий с вопросом пользователя
"""
        return context_message
    
    def _prepare_documents_context(self, query: str, dialog_history, retrieved_docs: str) -> str:
        """Подготовка контекста для DocumentAnalyst - только документы и история диалога"""
        formatted_history = self._format_dialog_history(dialog_history)
        
        context_message = f"""
АНАЛИЗ ДОКУМЕНТОВ И ЗНАНИЙ:

История диалога:
{formatted_history}

ДОКУМЕНТЫ И ГРАФОВЫЕ СВЯЗИ:
{retrieved_docs if retrieved_docs else "Документы не предоставлены"}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{query}

Проанализируй документы и найди теоретическую информацию, связанную с вопросом пользователя. Сосредоточься на:
- Механиках, правилах и ограничениях
- Информации о сохранениях и синхронизации данных
- Связях между различными компонентами системы
- Источниках каждого факта
"""
        return context_message
    
    def _prepare_integration_context(self, query: str, dialog_history, logs_analysis: str, docs_analysis: str) -> str:
        """Подготовка контекста для IntegrationSpecialist - результаты от других агентов"""
        formatted_history = self._format_dialog_history(dialog_history)
        
        context_message = f"""
ИНТЕГРАЦИЯ РЕЗУЛЬТАТОВ:

История диалога:
{formatted_history}

РЕЗУЛЬТАТЫ АНАЛИЗА ЛОГОВ:
{logs_analysis}

РЕЗУЛЬТАТЫ АНАЛИЗА ДОКУМЕНТОВ:
{docs_analysis}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{query}

Создай полный ответ, объединяя результаты анализа логов и документов. Сосредоточься на:
- Создании связного нарратива
- Объяснении противоречий, если они есть
- Указании источников информации
- Ссылках на предыдущие ответы при необходимости
"""
        return context_message
    
    async def _execute_agent_workflow(
        self, 
        query: str, 
        dialog_history = None,
        retrieved_docs: str = ""
    ) -> Dict[str, Any]:
        """Выполнение workflow агентов с новым подходом"""
        try:
            # Сохранение текущих параметров
            self.current_query = query
            self.current_dialog_history = dialog_history
            self.current_retrieved_docs = retrieved_docs
            self.ui_history = dialog_history.copy() if dialog_history else []
            
            # Старый общий контекст больше не используется - теперь каждый агент получает специализированный контекст
            
            # Выполняем новый workflow с invoke_stream
            return await self._execute_new_workflow(query, dialog_history, retrieved_docs)
            
        except Exception as e:
            logger.error(f"Ошибка в workflow агентов: {e}")
            return {
                "final_answer": f"Произошла ошибка: {str(e)}",
                "sources": [],
                "ui_history": self.ui_history,
                "error": str(e)
            }
    
    async def _execute_new_workflow(self, query: str, dialog_history, retrieved_docs: str) -> Dict[str, Any]:
        """Новый workflow с использованием invoke_stream и специализированных контекстов"""
        logger.info("🚀 Выполнение нового workflow со специализированными контекстами")
        
        final_response = ""
        logs_analysis = ""
        docs_analysis = ""
        
        # Шаг 1: LogsAnalyst - анализ только логов
        try:
            logger.info("🤖 Запуск агента: LogsAnalyst")
            logs_context = self._prepare_logs_context(query, dialog_history)
            
            full_response = []
            async for response in self.logs_agent.invoke_stream(
                messages=logs_context,
                thread=self.thread,
            ):
                self.thread = response.thread
                content_items = list(response.items)
                for item in content_items:
                    if hasattr(item, 'text') and item.text:
                        full_response.append(item.text)
            
            logs_analysis = ''.join(full_response)
            logger.info(f"✅ LogsAnalyst: {logs_analysis[:100]}...")
            
            self.ui_history.append({
                "role": "assistant",
                "content": logs_analysis,
                "metadata": {"agent": "LogsAnalyst", "title": "🗂️ Анализ логов"}
            })
            
        except Exception as e:
            logger.error(f"❌ Ошибка LogsAnalyst: {e}")
            logs_analysis = f"Ошибка анализа логов: {str(e)}"
            self.ui_history.append({
                "role": "assistant", 
                "content": logs_analysis,
                "metadata": {"agent": "LogsAnalyst", "title": "❌ Ошибка анализа логов"}
            })
        
        # Шаг 2: DocumentAnalyst - анализ только документов
        try:
            logger.info("🤖 Запуск агента: DocumentAnalyst")
            docs_context = self._prepare_documents_context(query, dialog_history, retrieved_docs)
            
            full_response = []
            async for response in self.rag_agent.invoke_stream(
                messages=docs_context,
                thread=self.thread,
            ):
                self.thread = response.thread
                content_items = list(response.items)
                for item in content_items:
                    if hasattr(item, 'text') and item.text:
                        full_response.append(item.text)
            
            docs_analysis = ''.join(full_response)
            logger.info(f"✅ DocumentAnalyst: {docs_analysis[:100]}...")
            
            self.ui_history.append({
                "role": "assistant",
                "content": docs_analysis,
                "metadata": {"agent": "DocumentAnalyst", "title": "📚 Анализ документов"}
            })
            
        except Exception as e:
            logger.error(f"❌ Ошибка DocumentAnalyst: {e}")
            docs_analysis = f"Ошибка анализа документов: {str(e)}"
            self.ui_history.append({
                "role": "assistant", 
                "content": docs_analysis,
                "metadata": {"agent": "DocumentAnalyst", "title": "❌ Ошибка анализа документов"}
            })
        
        # Шаг 3: IntegrationSpecialist - интеграция результатов
        try:
            logger.info("🤖 Запуск агента: IntegrationSpecialist")
            integration_context = self._prepare_integration_context(query, dialog_history, logs_analysis, docs_analysis)
            
            full_response = []
            async for response in self.integration_agent.invoke_stream(
                messages=integration_context,
                thread=self.thread,
            ):
                self.thread = response.thread
                content_items = list(response.items)
                for item in content_items:
                    if hasattr(item, 'text') and item.text:
                        full_response.append(item.text)
            
            final_response = ''.join(full_response)
            logger.info(f"✅ IntegrationSpecialist: {final_response[:100]}...")
            
            self.ui_history.append({
                "role": "assistant",
                "content": final_response,
                # "metadata": {"agent": "IntegrationSpecialist", "title": "🔗 Интеграция результатов"}
            })
            
        except Exception as e:
            logger.error(f"❌ Ошибка IntegrationSpecialist: {e}")
            final_response = f"Не удалось интегрировать результаты: {str(e)}"
            self.ui_history.append({
                "role": "assistant", 
                "content": final_response,
                "metadata": {"agent": "IntegrationSpecialist", "title": "❌ Ошибка интеграции"}
            })
        
        return {
            "final_answer": final_response,
            "sources": ["documents", "logs"],
            "ui_history": self.ui_history,
            "is_complete": True
        }
    
    def update_model(self, model_name: str):
        """Обновление модели для всех агентов"""
        if model_name != self.model_name:
            self.model_name = model_name
            self.llm = ClaudeCodeLLM(model_name=model_name)
            # Сброс агентов для пересоздания с новой моделью
            self.logs_agent = None
            self.rag_agent = None
            self.integration_agent = None
            self.thread = None
            logger.info(f"Multi-agent system updated to use model: {model_name}")
    
    @trace_function("MultiAgent_Processing")
    def process_query_with_streaming(
        self, 
        query: str, 
        dialog_history = None,
        retrieved_docs: str = ""
    ) -> Generator[List[dict], None, None]:
        """
        Streaming версия с использованием Semantic Kernel AgentGroupChat
        """
        try:
            # Добавляем контекст для LangSmith
            if is_tracing_enabled():
                logger.info(f"🔍 LangSmith: Starting multi-agent processing for query: {query[:100]}...")
            
            # Выполняем workflow асинхронно
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                workflow_results = loop.run_until_complete(
                    self._execute_agent_workflow(query, dialog_history, retrieved_docs)
                )
                
                # Возвращаем обновленную UI историю
                if workflow_results.get("ui_history"):
                    yield workflow_results["ui_history"]
                else:
                    # Fallback
                    final_history = (dialog_history.copy() if dialog_history else [])
                    final_history.append({
                        "role": "assistant",
                        "content": workflow_results.get("final_answer", "Не удалось получить ответ"),
                        "metadata": {"sources": workflow_results.get("sources", [])}
                    })
                    yield final_history
                    
            finally:
                loop.close()
            
        except Exception as e:
            logger.error(f"Ошибка в workflow streaming: {e}")
            error_history = (dialog_history.copy() if dialog_history else [])
            error_history.append({
                "role": "assistant",
                "content": f"Произошла ошибка: {str(e)}",
                "metadata": {"title": "❌ Ошибка"}
            })
            yield error_history