# Интеграция моделей в многоагентную RAG систему

## Обзор

Многоагентная система теперь поддерживает динамическую смену моделей из интерфейса пользователя. Система автоматически обновляет все агенты при изменении модели.

## Поток обработки модели

```
UI (выбор модели) → ModelManager → MultiAgentManager → Все агенты
```

### 1. Интерфейс пользователя
- Выбор модели из списка `MODELS` в `__init__.py`
- Передача выбранной модели в `generate_response_stream()`

### 2. ModelManager
- Получает параметр `model` из UI
- Передает модель в `multi_agent_manager.update_model()`
- Обновляет собственные агенты для fallback сценариев

### 3. MultiAgentManager  
- Метод `update_model()` обновляет:
  - `self.llm` для всех промптов
  - `self.sql_agent` с новой моделью
  - Логирует изменение модели

### 4. Агенты
- **Классификатор**: использует обновленный `self.llm`
- **RAG агент**: использует обновленный `self.llm`
- **SQL агент**: пересоздается с новой моделью
- **Интегратор**: использует обновленный `self.llm`

## Поддерживаемые модели

Система работает с любыми моделями из списка `MODELS`:

```python
MODELS: list = ["deepseek/deepseek-r1:free", "meta-llama/llama-3.3-70b-instruct:free"]
```

### Добавление новых моделей

1. **Обновите `__init__.py`:**
```python
MODELS: list = [
    "deepseek/deepseek-r1:free", 
    "meta-llama/llama-3.3-70b-instruct:free",
    "anthropic/claude-3-haiku:beta",  # Новая модель
    "openai/gpt-4o-mini"
]
```

2. **Проверьте совместимость:**
- Все модели должны поддерживать OpenRouter API
- Убедитесь в корректной работе с tool calling для SQL агента

## Оптимизация под разные модели

### Быстрые модели (gpt-4o-mini, claude-haiku)
- Подходят для классификации запросов
- Быстрая обработка простых RAG запросов
- Ограниченное качество комплексной интеграции

### Мощные модели (gpt-4, claude-3-opus)  
- Лучшая интеграция результатов
- Более точная классификация сложных запросов
- Выше стоимость запросов

### Специализированные модели
- **DeepSeek**: хорошо работает с кодом и техническими документами
- **LLaMA**: эффективен для аналитических задач
- **Claude**: отличная интеграция и рассуждения

## Конфигурация по умолчанию

```python
# В MultiAgentManager
def __init__(self, document_manager, database_url, model_name=None):
    # Если модель не указана, берем первую из списка
    if model_name is None:
        from __init__ import MODELS
        model_name = MODELS[0]
    
    self.model_name = model_name
    self.llm = ChatOpenRouter(model_name=model_name)
```

## Логирование и отладка

Система логирует все изменения модели:

```python
logger.info(f"Multi-agent system updated to use model: {model_name}")
logger.info(f"Using multi-agent system with model {model} [uid - {uid}]")
```

### Проверка активной модели
```python
print(f"Текущая модель: {multi_agent_manager.model_name}")
print(f"LLM модель: {multi_agent_manager.llm.model}")
```

## Обработка ошибок

### Недоступная модель
```python
try:
    self.llm = ChatOpenRouter(model_name=model_name)
except Exception as e:
    logger.error(f"Failed to initialize model {model_name}: {e}")
    # Fallback на модель по умолчанию
    self.llm = ChatOpenRouter(model_name=MODELS[0])
```

### Ошибка SQL агента
```python
try:
    self.sql_agent = self._create_sql_agent()
except Exception as e:
    logger.error(f"Failed to create SQL agent with model {model_name}: {e}")
    # Система продолжит работать только с RAG
```

## Производительность

### Кэширование агентов
Для оптимизации можно добавить кэширование:

```python
def update_model(self, model_name: str):
    if model_name == self.model_name:
        return  # Модель не изменилась
    
    # Сохраняем предыдущий агент для быстрого переключения
    self._previous_agents[self.model_name] = self.sql_agent
    
    if model_name in self._previous_agents:
        self.sql_agent = self._previous_agents[model_name]
    else:
        self.sql_agent = self._create_sql_agent()
```

### Предварительная инициализация
```python
def preload_models(self, models: List[str]):
    """Предзагрузка агентов для всех моделей"""
    for model in models:
        temp_llm = ChatOpenRouter(model_name=model)
        agent = create_sql_agent(temp_llm, db=self.sql_db, ...)
        self._model_cache[model] = agent
```

## Тестирование с разными моделями

```python
# Автоматическое тестирование всех моделей
from __init__ import MODELS

for model in MODELS:
    print(f"Тестируем модель: {model}")
    multi_agent.update_model(model)
    
    result = multi_agent.process_query_sync("Тестовый запрос", f"test_{model}")
    print(f"Результат: {result['query_type']}")
```

## Рекомендации

### Для продакшена
1. **Используйте стабильные модели** для критических задач
2. **Мониторьте стоимость** запросов при смене моделей
3. **Тестируйте качество** ответов на реальных данных
4. **Настройте fallback** на проверенную модель

### Для разработки
1. **Используйте быстрые модели** для итераций
2. **Тестируйте с разными моделями** для оценки качества
3. **Логируйте результаты** для сравнения производительности

### Для специфичных задач
- **Технические документы**: DeepSeek, CodeLLaMA
- **Аналитика**: Claude, GPT-4
- **Быстрые ответы**: GPT-4o-mini, Claude Haiku
- **Мультимодальность**: GPT-4V, Claude 3