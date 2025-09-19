"""
Конфигурация LangSmith для мониторинга и отладки LocalGPT
"""
import os
from langsmith import Client
from typing import Optional

class LangSmithConfig:
    """Конфигурация LangSmith для трейсинга и мониторинга"""
    
    def __init__(self):
        # Настройки LangSmith (установите переменные окружения или укажите здесь)
        self.api_key = os.getenv("LANGSMITH_API_KEY", "")
        self.endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
        self.project_name = os.getenv("LANGSMITH_PROJECT", "LocalGPT-MultiAgent")
        
        # Включение/выключение трейсинга
        self.tracing_enabled = bool(self.api_key)
        
        if self.tracing_enabled:
            # Установка переменных окружения для автоматического трейсинга
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = self.endpoint
            os.environ["LANGCHAIN_API_KEY"] = self.api_key
            os.environ["LANGCHAIN_PROJECT"] = self.project_name
            
            # Создание клиента
            self.client = Client(
                api_url=self.endpoint,
                api_key=self.api_key
            )
        else:
            self.client = None
    
    def is_enabled(self) -> bool:
        """Проверка, включен ли трейсинг"""
        return self.tracing_enabled
    
    def get_client(self) -> Optional[Client]:
        """Получение клиента LangSmith"""
        return self.client

# Глобальная конфигурация
langsmith_config = LangSmithConfig()

def get_langsmith_client() -> Optional[Client]:
    """Получение клиента LangSmith"""
    return langsmith_config.get_client()

def is_tracing_enabled() -> bool:
    """Проверка, включен ли трейсинг"""
    return langsmith_config.is_enabled()

# Декораторы для трейсинга
def trace_function(name: str = None):
    """Декоратор для трейсинга функций"""
    def decorator(func):
        if not is_tracing_enabled():
            return func
            
        from langsmith import traceable
        return traceable(name=name or func.__name__)(func)
    return decorator

def trace_agent(agent_name: str):
    """Декоратор для трейсинга агентов"""
    def decorator(func):
        if not is_tracing_enabled():
            return func
            
        from langsmith import traceable
        return traceable(name=f"Agent_{agent_name}")(func)
    return decorator