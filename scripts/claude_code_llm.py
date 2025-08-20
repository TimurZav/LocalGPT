"""
Обертка для Claude Code SDK для совместимости с LangChain интерфейсом
"""
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from claude_code_sdk import query, ClaudeCodeOptions, AssistantMessage, TextBlock
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import LLMResult, Generation
from langchain_core.callbacks.manager import CallbackManagerForLLMRun


@dataclass 
class ClaudeCodeResponse:
    """Результат запроса к Claude Code"""
    content: str
    
    
class ClaudeCodeLLM(LLM):
    """LangChain-совместимая обертка для Claude Code SDK"""
    
    model_name: str = "claude-3-5-sonnet-20241022"
    max_turns: int = 1
    
    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", max_turns: int = 1, **kwargs):
        super().__init__(model_name=model_name, max_turns=max_turns, **kwargs)
        
    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> tuple[str, Optional[str]]:
        """Конвертирует LangChain сообщения в промпт для Claude Code"""
        system_prompt = None
        user_messages = []
        
        for message in messages:
            if isinstance(message, SystemMessage):
                system_prompt = message.content
            elif isinstance(message, HumanMessage):
                user_messages.append(message.content)
            elif isinstance(message, AIMessage):
                # Добавляем AI сообщения как контекст
                user_messages.append(f"Assistant: {message.content}")
        
        # Объединяем все пользовательские сообщения
        prompt = "\n\n".join(user_messages)
        
        return prompt, system_prompt
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Основной метод для LLM - выполняет запрос к Claude Code"""
        # Создаем опции для Claude Code
        options = ClaudeCodeOptions(
            system_prompt=kwargs.get('system_prompt'),
            max_turns=self.max_turns
        )
        
        # Синхронный запрос с правильной обработкой event loop
        async def _run_query():
            response_text = ""
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text
            return response_text
        
        try:
            # Проверяем, есть ли уже запущенный event loop
            loop = asyncio.get_running_loop()
            # Если есть, создаем новый поток для выполнения
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _run_query())
                return future.result()
        except RuntimeError:
            # Нет запущенного event loop, можем использовать asyncio.run
            return asyncio.run(_run_query())
    
    async def _agenerate(
        self,
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
        run_manager = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Асинхронная генерация ответов"""
        generations = []
        
        for message_list in messages:
            prompt, system_prompt = self._convert_messages_to_prompt(message_list)
            
            # Создаем опции для Claude Code
            options = ClaudeCodeOptions(
                system_prompt=system_prompt,
                max_turns=self.max_turns
            )
            
            # Выполняем запрос
            response_text = ""
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text
            
            generations.append([Generation(text=response_text)])
        
        return LLMResult(generations=generations)
    
    def _generate(
        self,
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
        run_manager = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Синхронная генерация ответов"""
        try:
            # Проверяем, есть ли уже запущенный event loop
            loop = asyncio.get_running_loop()
            # Если есть, создаем новый поток для выполнения
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self._agenerate(messages, stop, run_manager, **kwargs))
                return future.result()
        except RuntimeError:
            # Нет запущенного event loop, можем использовать asyncio.run
            return asyncio.run(self._agenerate(messages, stop, run_manager, **kwargs))
    
    def invoke(self, messages: List[BaseMessage], **kwargs) -> ClaudeCodeResponse:
        """Простой интерфейс для вызова (совместимость с OpenRouter)"""
        prompt, system_prompt = self._convert_messages_to_prompt(messages)
        
        # Используем _call метод с system_prompt
        response_text = self._call(prompt, system_prompt=system_prompt, **kwargs)
        return ClaudeCodeResponse(content=response_text)
    
    @property
    def _llm_type(self) -> str:
        """Возвращает тип LLM"""
        return "claude-code"
        
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Параметры для идентификации модели"""
        return {
            "model_name": self.model_name,
            "max_turns": self.max_turns
        }