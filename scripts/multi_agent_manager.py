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
from openai import AsyncOpenAI
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread, AgentGroupChat
from semantic_kernel.agents.strategies import (
    SequentialSelectionStrategy,
)
from semantic_kernel.agents.strategies.termination.termination_strategy import TerminationStrategy
from pydantic import Field
import re
from semantic_kernel.kernel import Kernel
from semantic_kernel.contents import AuthorRole, ChatMessageContent

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


class SimpleApprovedTerminationStrategy(TerminationStrategy):
    """Simple termination strategy that looks for 'approved' in messages"""
    
    current_iterations: int = Field(default=0)
    
    def __init__(self, maximum_iterations: int = 3, **data):
        super().__init__(maximum_iterations=maximum_iterations, **data)
    
    async def should_terminate(self, agent, history) -> bool:
        """Check if conversation should terminate"""
        self.current_iterations += 1
        
        # Check max iterations first
        if self.current_iterations >= self.maximum_iterations:
            return True
        
        # Check last message for positive 'approved' (not "not approved")
        if history and len(history) > 0:
            last_message = history[-1]
            if hasattr(last_message, 'content') and last_message.content:
                content = last_message.content.lower()
                # Look for "approved" but not preceded by "not" 
                if re.search(r'(?<!not\s)(?<!not\s\w\s)approved', content) and 'not approved' not in content:
                    return True
        
        return False


def _create_kernel_with_chat_completion() -> Kernel:
    kernel = Kernel()

    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://models.inference.ai.azure.com/",
    )

    kernel.add_service(
        OpenAIChatCompletion(
            ai_model_id="gpt-4o-mini",
            async_client=client,
        )
    )

    return kernel


class MultiAgentManager:
    """Менеджер многоагентной системы на основе Semantic Kernel ChatCompletionAgent"""
    
    def __init__(self, document_manager, model_name: str = "claude-3-5-sonnet-20241022"):
        self.document_manager = document_manager
        self.model_name = model_name
        self.llm = ClaudeCodeLLM(model_name=model_name)
        
        # Агенты будут созданы при первом запросе
        self.logs_agent = None
        self.rag_agent = None
        self.integration_agent = None
        self.edd_agent = None  # Evaluation-Driven Delivery агент
        self.goal_delivery_agent = None  # Goal Delivery агент
        self.quality_reviewer = None  # Quality Reviewer агент
        self.thread = None  # ChatHistoryAgentThread для управления диалогом
        
        # Дополнительное состояние для новых агентов
        self.project_goals = []  # История целей проекта
        self.evaluation_history = []  # История оценок и тестов
        self.skills_progress = {}  # Прогресс навыков разработчика
        
        # Текущее состояние обработки
        self.current_query = None
        self.current_dialog_history = None
        self.current_retrieved_docs = None
        self.ui_history = []
    
    @staticmethod
    def _create_logs_agent() -> ChatCompletionAgent:
        """Создание агента анализа логов"""
        return ChatCompletionAgent(
            kernel=_create_kernel_with_chat_completion(),
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
    
    @staticmethod
    def _create_rag_agent() -> ChatCompletionAgent:
        """Создание RAG агента для работы с документами"""
        return ChatCompletionAgent(
            kernel=_create_kernel_with_chat_completion(),
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
    
    @staticmethod
    def _create_integration_agent() -> ChatCompletionAgent:
        """Создание интеграционного агента"""
        return ChatCompletionAgent(
            kernel=_create_kernel_with_chat_completion(),
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
    def _create_edd_agent() -> ChatCompletionAgent:
        """Создание EDD (Evaluation-Driven Delivery) агента"""
        return ChatCompletionAgent(
            kernel=_create_kernel_with_chat_completion(),
            name="EDDSpecialist",
            instructions="""Ты EDD (Evaluation-Driven Delivery) специалист. Твоя задача - обеспечить постоянную проверку прогресса к цели на основе evaluations.

            Ключевые принципы EDD:
            1. Каждое изменение должно быть проверяемо через тесты/evals
            2. Прогресс измеряется конкретными метриками, а не субъективными оценками
            3. Быстрая обратная связь через автоматизированные проверки
            4. Инкрементальная доставка ценности

            Твои задачи:
            - Анализировать требования и предлагать конкретные критерии успеха
            - Создавать план тестирования для каждой функции
            - Проверять актуальность существующих тестов при изменении требований  
            - Выявлять пробелы в покрытии тестами
            - Предлагать метрики для измерения прогресса
            - Рекомендовать инструменты для автоматизации оценок

            Формат ответа:
            ## Анализ требований
            [Анализ текущих требований]
            
            ## Критерии успеха
            [Конкретные, измеримые критерии]
            
            ## План тестирования
            [Предложения по тестам и evals]
            
            ## Рекомендации
            [Следующие шаги для улучшения процесса]"""
        )
    
    @staticmethod
    def _create_goal_delivery_agent() -> ChatCompletionAgent:
        """Создание Goal Delivery агента для управления целями и прогрессом"""
        return ChatCompletionAgent(
            kernel=_create_kernel_with_chat_completion(),
            name="GoalDeliveryManager",
            instructions="""Ты Goal Delivery Manager - агент, отвечающий за интеграцию работы всех агентов для успешного завершения проектов.

            Твои основные функции:
            1. КОНТРОЛЬ ВЫПОЛНЕНИЯ: Отслеживаешь выполнение конечных требований
            2. УПРАВЛЕНИЕ ТЕСТАМИ: Превращаешь требования в тесты, следишь за их актуальностью
            3. ПРОГРЕСС-ТРЕКИНГ: Накапливаешь историю выполненных целей и прогресса
            4. МЕНТОРСТВО: Фиксируешь приобретаемые навыки и ошибки разработчика
            5. КООРДИНАЦИЯ: Интегрируешь результаты всех других агентов

            Ключевые принципы:
            - Всё задуманное должно быть реализовано и проверено тестами
            - Требования → Тесты → Реализация → Проверка
            - Непрерывное обучение через анализ ошибок и успехов
            - Прозрачность прогресса для всех участников

            Формат ответа:
            ## Статус целей
            [Анализ выполнения текущих целей]
            
            ## План действий
            [Конкретные шаги для достижения целей]
            
            ## Тестовое покрытие
            [Анализ покрытия требований тестами]
            
            ## Обучающие выводы
            [Навыки, ошибки, рекомендации для развития]
            
            ## Следующие шаги
            [Приоритетные задачи и действия]"""
        )
    
    @staticmethod
    def _create_quality_reviewer_agent() -> ChatCompletionAgent:
        """Создание агента проверки качества ответов других агентов"""
        return ChatCompletionAgent(
            kernel=_create_kernel_with_chat_completion(),
            name="QualityReviewer",
            instructions="""Ты Quality Reviewer - агент, который проверяет качество ответов других агентов.

            Твоя задача - оценить, соответствует ли ответ агента поставленной задаче и стандартам качества.

            Критерии оценки:
            1. РЕЛЕВАНТНОСТЬ: Отвечает ли агент на заданный вопрос?
            2. ПОЛНОТА: Достаточно ли информативен ответ?
            3. СТРУКТУРА: Хорошо ли структурирован ответ?
            4. СПЕЦИАЛИЗАЦИЯ: Соответствует ли ответ роли агента?
            5. ПРАКТИЧНОСТЬ: Можно ли использовать информацию на практике?

            Если ответ ХОРОШЕГО КАЧЕСТВА, ответь: "APPROVED: [краткое обоснование]"
            Если ответ ТРЕБУЕТ УЛУЧШЕНИЯ, ответь: "NEEDS_IMPROVEMENT: [конкретные рекомендации]"

            Будь конструктивен и четок в своих оценках. Указывай конкретные проблемы и пути их решения.
            
            Примеры:
            - "APPROVED: Ответ полный, структурированный и отвечает на все аспекты вопроса"
            - "NEEDS_IMPROVEMENT: Ответ слишком общий, добавь конкретные примеры и структурируй по пунктам"
            - "NEEDS_IMPROVEMENT: Не учтен контекст логов, проанализируй связь с техническими проблемами"
            """
        )
    
    @staticmethod
    def _create_thread() -> ChatHistoryAgentThread:
        """Создание thread для управления диалогом между агентами"""
        return ChatHistoryAgentThread()
    
    def _initialize_agents(self):
        """Инициализация всех агентов и thread для управления диалогом"""
        try:
            # Создание основных агентов
            if not self.logs_agent:
                self.logs_agent = self._create_logs_agent()
            if not self.rag_agent:
                self.rag_agent = self._create_rag_agent()
            if not self.integration_agent:
                self.integration_agent = self._create_integration_agent()
                
            # Создание новых агентов
            if not self.edd_agent:
                self.edd_agent = self._create_edd_agent()
            if not self.goal_delivery_agent:
                self.goal_delivery_agent = self._create_goal_delivery_agent()
            if not self.quality_reviewer:
                self.quality_reviewer = self._create_quality_reviewer_agent()
            
            # Создание thread для управления диалогом
            if not self.thread:
                self.thread = self._create_thread()
            
            logger.info("✅ Все агенты и thread инициализированы")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации агентов: {e}")
            raise e
    
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
    
    async def _execute_agent_with_review_cycle(
        self,
        agent,
        agent_name: str,
        context: str,
        max_iterations: int = 3
    ) -> tuple[str, bool]:
        """
        Выполняет агента с циклом проверки через QualityReviewer (по принципу из документации)
        
        Args:
            agent: Агент для выполнения
            agent_name: Имя агента
            context: Контекст для агента
            max_iterations: Максимальное количество итераций
        
        Returns:
            tuple[str, bool]: (финальный ответ, успешность)
        """
        
        logger.info(f"🔄 Запуск цикла проверки для {agent_name}")
        
        try:
            # Создаем AgentGroupChat по принципу из документации
            chat = AgentGroupChat(
                agents=[agent, self.quality_reviewer],
                termination_strategy=SimpleApprovedTerminationStrategy(maximum_iterations=max_iterations),
                selection_strategy=SequentialSelectionStrategy(),
            )

            # Добавляем начальное сообщение пользователя
            await chat.add_chat_message(ChatMessageContent(role=AuthorRole.USER, content=context))
            logger.info(f"# User: '{context[:100]}...'")
            
            final_response = ""
            
            # Запускаем цикл общения агентов
            async for content in chat.invoke():
                logger.info(f"# Agent - {content.name or '*'}: '{content.content[:100]}...'")
                
                # Добавляем ВСЕ ответы агентов в UI для прозрачности
                if content.name and content.content:
                    # Определяем иконку и заголовок для каждого типа агента
                    agent_titles = {
                        "LogsAnalyst": "🗂️ Анализ логов",
                        "DocumentAnalyst": "📚 Анализ документов", 
                        "IntegrationSpecialist": "🔗 Интеграция результатов",
                        "EDDSpecialist": "🧪 EDD Анализ",
                        "GoalDeliveryManager": "🎯 Управление целями",
                        "QualityReviewer": "🔍 Проверка качества"
                    }
                    
                    title = agent_titles.get(content.name, f"🤖 {content.name}")
                    
                    self.ui_history.append({
                        "role": "assistant",
                        "content": content.content,
                        "metadata": {
                            "agent": content.name,
                            "title": title,
                            "is_review_cycle": True,
                            "parent_agent": agent_name if content.name != agent_name else None
                        }
                    })
                
                # Сохраняем последний ответ от основного агента для возврата
                if content.name == agent_name:
                    final_response = content.content

            logger.info(f"# IS COMPLETE: {chat.is_complete}")
            
            return final_response, chat.is_complete
            
        except Exception as e:
            logger.error(f"❌ Ошибка в цикле проверки {agent_name}: {e}")
            # Fallback: выполняем агента без проверки
            logger.info(f"🔄 Fallback: запуск {agent_name} без проверки")
            try:
                full_response = []
                async for response in agent.invoke_stream(
                    messages=context,
                    thread=self.thread,
                ):
                    self.thread = response.thread
                    content_items = list(response.items)
                    for item in content_items:
                        if hasattr(item, 'text') and item.text:
                            full_response.append(item.text)
                
                return ''.join(full_response), True
                
            except Exception as e2:
                logger.error(f"❌ Критическая ошибка в {agent_name}: {e2}")
                return f"Ошибка выполнения {agent_name}: {str(e2)}", False
    
    async def _execute_agent_with_retry(
        self, 
        agent, 
        agent_name: str, 
        context: str, 
        max_retries: int = 2,
        quality_check_fn=None
    ) -> tuple[str, bool]:
        """
        Выполнить агента с возможностью повторных попыток
        
        Args:
            agent: Агент для выполнения
            agent_name: Имя агента для логирования
            context: Контекст для агента
            max_retries: Максимальное количество попыток
            quality_check_fn: Функция проверки качества ответа
            
        Returns:
            tuple[str, bool]: (ответ агента, успешность выполнения)
        """
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"🤖 Запуск агента: {agent_name} (попытка {attempt + 1}/{max_retries + 1})")
                
                full_response = []
                async for response in agent.invoke_stream(
                    messages=context,
                    thread=self.thread,
                ):
                    self.thread = response.thread
                    content_items = list(response.items)
                    for item in content_items:
                        if hasattr(item, 'text') and item.text:
                            full_response.append(item.text)
                
                agent_response = ''.join(full_response)
                
                # Проверка качества ответа, если задана функция проверки
                if quality_check_fn:
                    quality_score, feedback = quality_check_fn(agent_response, context)
                    if quality_score < 0.7:  # Порог качества
                        if attempt < max_retries:
                            logger.warning(f"⚠️ {agent_name}: Качество ответа ниже порога ({quality_score:.2f}). Повторяю попытку...")
                            logger.info(f"📝 Обратная связь: {feedback}")
                            # Обновляем контекст с обратной связью
                            context += f"\n\nОБРАТНАЯ СВЯЗЬ (предыдущая попытка была неудовлетворительной):\n{feedback}\nПожалуйста, улучши ответ с учетом этой обратной связи."
                            continue
                        else:
                            logger.warning(f"⚠️ {agent_name}: Исчерпаны попытки, используем последний ответ")
                
                logger.info(f"✅ {agent_name}: {agent_response[:100]}...")
                return agent_response, True
                
            except Exception as e:
                logger.error(f"❌ Ошибка {agent_name} (попытка {attempt + 1}): {e}")
                if attempt < max_retries:
                    logger.info(f"🔄 Повторяю попытку для {agent_name}...")
                    continue
                else:
                    error_message = f"Ошибка в работе агента {agent_name} после {max_retries + 1} попыток: {str(e)}"
                    return error_message, False
        
        return "Неизвестная ошибка", False
    
    @staticmethod
    def _check_response_quality(response: str, context: str) -> tuple[float, str]:
        """
        Проверка качества ответа агента
        
        Args:
            response: Ответ агента
            context: Контекст запроса
            
        Returns:
            tuple[float, str]: (оценка качества 0-1, обратная связь)
        """
        quality_score = 0.5  # Базовая оценка
        feedback_items = []
        
        # Проверки качества
        if len(response) < 50:
            feedback_items.append("Ответ слишком короткий и неинформативный")
        else:
            quality_score += 0.2
            
        if "ошибка" in response.lower() and "ошибка" in response.lower()[:100]:
            feedback_items.append("Ответ начинается с ошибки, попробуй дать более конструктивный ответ")
        else:
            quality_score += 0.15
            
        if any(marker in response for marker in ["##", "- ", "1.", "2."]):
            quality_score += 0.15  # Структурированный ответ
        else:
            feedback_items.append("Добавь структуру в ответ (заголовки, списки, нумерацию)")
            
        # Проверка на релевантность (базовая)
        query_words = context.lower().split()[:20]  # Первые 20 слов контекста
        response_words = response.lower().split()
        common_words = set(query_words) & set(response_words)
        
        if len(common_words) >= 3:
            quality_score += 0.1
        else:
            feedback_items.append("Ответ не связан с вопросом, будь более релевантным")
        
        feedback = "; ".join(feedback_items) if feedback_items else "Ответ соответствует требованиям"
        return min(quality_score, 1.0), feedback
    
    def _prepare_edd_context(self, query: str, dialog_history, logs_analysis: str, docs_analysis: str) -> str:
        """Подготовка контекста для EDD агента"""
        formatted_history = self._format_dialog_history(dialog_history)
        
        # Информация о текущих тестах и evaluations
        current_evals = "\n".join(self.evaluation_history[-5:]) if self.evaluation_history else "История evaluations пуста"
        
        context_message = f"""
EVALUATION-DRIVEN DELIVERY АНАЛИЗ:

История диалога:
{formatted_history}

РЕЗУЛЬТАТЫ АНАЛИЗА ЛОГОВ:
{logs_analysis}

РЕЗУЛЬТАТЫ АНАЛИЗА ДОКУМЕНТОВ:
{docs_analysis}

ТЕКУЩИЕ EVALUATIONS И ТЕСТЫ:
{current_evals}

ВОПРОС/ТРЕБОВАНИЕ ПОЛЬЗОВАТЕЛЯ:
{query}

Проанализируй требования и создай план для EDD подхода. Сосредоточься на:
- Превращении требований в измеримые критерии успеха
- Создании плана тестирования и evaluations
- Выявлении пробелов в текущем тестовом покрытии
- Рекомендациях по инструментам автоматизации
"""
        return context_message
    
    def _prepare_goal_delivery_context(
        self, 
        query: str, 
        dialog_history, 
        logs_analysis: str, 
        docs_analysis: str, 
        edd_analysis: str
    ) -> str:
        """Подготовка контекста для Goal Delivery агента"""
        formatted_history = self._format_dialog_history(dialog_history)
        
        # Информация о целях проекта
        current_goals = "\n".join([f"- {goal}" for goal in self.project_goals[-3:]]) if self.project_goals else "Цели проекта не определены"
        
        # Прогресс навыков
        if self.skills_progress:
            skills_items = []
            for skill, data in self.skills_progress.items():
                level = data.get('level', 'начинающий')
                progress = data.get('progress', 0)
                skills_items.append(f"- {skill}: {level} ({progress}% прогресс)")
            skills_summary = "\n".join(skills_items)
        else:
            skills_summary = "Прогресс навыков не отслеживается"
        
        context_message = f"""
GOAL DELIVERY MANAGEMENT:

История диалога:
{formatted_history}

РЕЗУЛЬТАТЫ АНАЛИЗА ЛОГОВ:
{logs_analysis}

РЕЗУЛЬТАТЫ АНАЛИЗА ДОКУМЕНТОВ:
{docs_analysis}

EDD АНАЛИЗ И РЕКОМЕНДАЦИИ:
{edd_analysis}

ТЕКУЩИЕ ЦЕЛИ ПРОЕКТА:
{current_goals}

ПРОГРЕСС НАВЫКОВ РАЗРАБОТЧИКА:
{skills_summary}

ВОПРОС/ТРЕБОВАНИЕ ПОЛЬЗОВАТЕЛЯ:
{query}

Интегрируй все результаты и создай план управления целями. Сосредоточься на:
- Контроле выполнения требований
- Управлении тестами и их актуальности
- Отслеживании прогресса к целям
- Выявлении обучающих моментов для разработчика
- Координации работы всех агентов
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
            # Инициализация агентов при первом запросе
            if not self.logs_agent:
                self._initialize_agents()
                
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
        """Новый 5-агентный workflow с повторными попытками"""
        logger.info("🚀 Выполнение расширенного workflow с 5 агентами")
        
        # Шаг 1: LogsAnalyst - анализ только логов
        logs_context = self._prepare_logs_context(query, dialog_history)
        logs_analysis, success = await self._execute_agent_with_review_cycle(
            self.logs_agent, "LogsAnalyst", logs_context, max_iterations=4
        )
        
        # Шаг 2: DocumentAnalyst - анализ только документов
        docs_context = self._prepare_documents_context(query, dialog_history, retrieved_docs)
        docs_analysis, success = await self._execute_agent_with_review_cycle(
            self.rag_agent, "DocumentAnalyst", docs_context, max_iterations=4
        )
        
        # Шаг 3: IntegrationSpecialist - базовая интеграция
        integration_context = self._prepare_integration_context(query, dialog_history, logs_analysis, docs_analysis)
        integration_result, success = await self._execute_agent_with_review_cycle(
            self.integration_agent, "IntegrationSpecialist", integration_context, max_iterations=4
        )
        
        # Шаг 4: EDD Specialist - анализ для Evaluation-Driven Delivery
        edd_context = self._prepare_edd_context(query, dialog_history, logs_analysis, docs_analysis)
        edd_result, success = await self._execute_agent_with_review_cycle(
            self.edd_agent, "EDDSpecialist", edd_context, max_iterations=4
        )
        
        # Обновляем историю evaluations
        if "## План тестирования" in edd_result:
            self.evaluation_history.append(f"Query: {query} -> EDD: {edd_result[:200]}...")
        
        # Шаг 5: Goal Delivery Manager - финальная координация
        goal_context = self._prepare_goal_delivery_context(query, dialog_history, logs_analysis, docs_analysis, edd_result)
        final_response, success = await self._execute_agent_with_review_cycle(
            self.goal_delivery_agent, "GoalDeliveryManager", goal_context, max_iterations=4
        )
        
        # Обновляем состояние системы на основе результатов Goal Delivery Manager
        self._update_system_state_from_goal_manager(final_response, query)
        
        return {
            "final_answer": final_response,
            "sources": ["documents", "logs", "evaluations", "goals"],
            "ui_history": self.ui_history,
            "is_complete": True,
            "agent_results": {
                "logs": logs_analysis,
                "docs": docs_analysis, 
                "integration": integration_result,
                "edd": edd_result,
                "goal_delivery": final_response
            }
        }
    
    def _update_system_state_from_goal_manager(self, response: str, query: str):
        """Обновление состояния системы на основе ответа Goal Delivery Manager"""
        try:
            # Извлекаем цели из ответа
            if "цель" in response.lower() or "требование" in response.lower():
                self.project_goals.append(f"{query} (из диалога)")
                # Ограничиваем историю последними 10 целями
                self.project_goals = self.project_goals[-10:]
            
            # Извлекаем информацию о навыках
            if "навык" in response.lower() or "обучение" in response.lower():
                # Простая эвристика для извлечения навыков
                if "python" in response.lower():
                    if "python" not in self.skills_progress:
                        self.skills_progress["python"] = {"level": "начинающий", "progress": 10}
                    else:
                        self.skills_progress["python"]["progress"] = min(100, self.skills_progress["python"]["progress"] + 5)
                        
                if "тест" in response.lower() or "test" in response.lower():
                    if "testing" not in self.skills_progress:
                        self.skills_progress["testing"] = {"level": "начинающий", "progress": 15}
                    else:
                        self.skills_progress["testing"]["progress"] = min(100, self.skills_progress["testing"]["progress"] + 10)
            
            logger.info(f"🔄 Обновлено состояние: {len(self.project_goals)} целей, {len(self.skills_progress)} навыков")
            
        except Exception as e:
            logger.warning(f"⚠️ Ошибка обновления состояния: {e}")
    
    def update_model(self, model_name: str):
        """Обновление модели для всех агентов"""
        if model_name != self.model_name:
            self.model_name = model_name
            self.llm = ClaudeCodeLLM(model_name=model_name)
            # Сброс всех агентов для пересоздания с новой моделью
            self.logs_agent = None
            self.rag_agent = None
            self.integration_agent = None
            self.edd_agent = None
            self.goal_delivery_agent = None
            self.quality_reviewer = None
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
