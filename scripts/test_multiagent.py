"""
Тестирование многоагентной системы
"""

import asyncio
import logging
from multi_agent_manager import MultiAgentManager
from app import DocumentManager
from __init__ import DATABASE_DATA_URL

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_multiagent_system():
    """Тест многоагентной системы"""
    
    # Инициализация менеджеров
    document_manager = DocumentManager()
    document_manager.db = document_manager.initialize_database()
    
    multi_agent = MultiAgentManager(
        document_manager=document_manager,
        database_url=DATABASE_DATA_URL,
        model_name="openai/gpt-4o-mini"
    )
    
    # Тестовые запросы
    test_queries = [
        {
            "query": "Какие требования к паролям установлены в компании?",
            "expected_type": "rag_only",
            "description": "RAG-only запрос (только документы)"
        },
        {
            "query": "Сколько сотрудников работает в отделе разработки?",
            "expected_type": "sql_only", 
            "description": "SQL-only запрос (только БД)"
        },
        {
            "query": "Какие проекты по безопасности ведутся в компании и какие требования безопасности должны соблюдаться?",
            "expected_type": "hybrid",
            "description": "Гибридный запрос (документы + БД)"
        },
        {
            "query": "Кто работает над проектом 'Умный город' и какие технические требования к этому проекту?",
            "expected_type": "hybrid",
            "description": "Гибридный запрос (команда проекта + техтребования)"
        }
    ]
    
    print("🧪 Тестирование многоагентной системы\n")
    
    for i, test in enumerate(test_queries, 1):
        print(f"{'='*50}")
        print(f"ТЕСТ {i}: {test['description']}")
        print(f"Ожидаемый тип: {test['expected_type']}")
        print(f"Вопрос: {test['query']}")
        print(f"{'='*50}")
        
        try:
            # Выполнение запроса
            result = await multi_agent.process_query(test['query'], f"test_{i}")
            
            print(f"✅ РЕЗУЛЬТАТ:")
            print(f"   Тип запроса: {result['query_type']}")
            print(f"   Ответ: {result['answer'][:200]}...")
            print(f"   Источники: {result['sources']}")
            
            # Проверка соответствия ожидаемому типу
            if result['query_type'] == test['expected_type']:
                print(f"   ✅ Классификация корректна")
            else:
                print(f"   ⚠️ Ожидался {test['expected_type']}, получен {result['query_type']}")
            
            print()
            
        except Exception as e:
            print(f"❌ ОШИБКА: {e}")
            print()

def test_sync_version():
    """Синхронная версия тестирования"""
    print("🧪 Синхронное тестирование\n")
    
    # Инициализация
    document_manager = DocumentManager()
    document_manager.db = document_manager.initialize_database()
    
    multi_agent = MultiAgentManager(
        document_manager=document_manager,
        database_url=DATABASE_DATA_URL,
        model_name="openai/gpt-4o-mini"
    )
    
    # Простой тест
    query = "Какие сотрудники работают в отделе разработки?"
    print(f"Тестовый запрос: {query}")
    
    try:
        result = multi_agent.process_query_sync(query, "sync_test")
        
        print(f"Тип запроса: {result['query_type']}")
        print(f"Ответ: {result['answer']}")
        print(f"Источники: {result['sources']}")
        
        if result['sql_result']:
            print(f"SQL результат успешен: {result['sql_result'].success}")
        if result['rag_result']:
            print(f"RAG результат успешен: {result['rag_result'].success}")
            
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    print("Выберите тип тестирования:")
    print("1. Асинхронное (полное)")
    print("2. Синхронное (быстрое)")
    
    choice = input("Введите номер (1 или 2): ").strip()
    
    if choice == "1":
        asyncio.run(test_multiagent_system())
    elif choice == "2":
        test_sync_version()
    else:
        print("Неверный выбор. Запуск синхронного тестирования...")
        test_sync_version()