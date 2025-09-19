"""
Оценки и метрики для LocalGPT с использованием LangSmith
"""
from typing import Dict, Any, List, Optional
from langsmith_config import get_langsmith_client, is_tracing_enabled
from langsmith.evaluation import evaluate, EvaluationResult
from langsmith.schemas import Example, Run
import logging

logger = logging.getLogger(__name__)

def relevance_evaluator(run: Run, example: Example) -> EvaluationResult:
    """
    Оценивает релевантность ответа на основе качества информации
    """
    inputs = run.inputs
    outputs = run.outputs
    
    # Получаем ответ модели
    answer = outputs.get("content", "") if outputs else ""
    question = inputs.get("query", "") if inputs else ""
    
    # Простая эвристика для оценки релевантности
    score = 0.0
    feedback = []
    
    # Проверяем длину ответа
    if len(answer) < 50:
        score += 0.2
        feedback.append("Ответ слишком короткий")
    elif len(answer) > 100:
        score += 0.8
        feedback.append("Ответ содержательный")
    
    # Проверяем наличие источников
    if "источник" in answer.lower() or "документ" in answer.lower():
        score += 0.2
        feedback.append("Указаны источники")
    
    # Проверяем связность с вопросом
    question_words = set(question.lower().split())
    answer_words = set(answer.lower().split())
    overlap = len(question_words.intersection(answer_words))
    
    if overlap > 0:
        score += min(0.3, overlap * 0.1)
        feedback.append(f"Совпадение ключевых слов: {overlap}")
    
    return EvaluationResult(
        key="relevance",
        score=min(1.0, score),
        comment=" | ".join(feedback)
    )

def completeness_evaluator(run: Run, example: Example) -> EvaluationResult:
    """
    Оценивает полноту ответа
    """
    outputs = run.outputs
    answer = outputs.get("content", "") if outputs else ""
    
    score = 0.0
    feedback = []
    
    # Проверяем структуру ответа
    if "##" in answer or "**" in answer:
        score += 0.3
        feedback.append("Структурированный ответ")
    
    # Проверяем наличие практических данных из логов
    if "лог" in answer.lower() or "событие" in answer.lower():
        score += 0.3
        feedback.append("Включены данные из логов")
    
    # Проверяем наличие теоретической информации
    if "документ" in answer.lower() or "согласно" in answer.lower():
        score += 0.3
        feedback.append("Включена теоретическая информация")
    
    # Проверяем наличие заключения/выводов
    if any(word in answer.lower() for word in ["итого", "вывод", "заключение", "резюмируя"]):
        score += 0.1
        feedback.append("Есть выводы")
    
    return EvaluationResult(
        key="completeness", 
        score=score,
        comment=" | ".join(feedback)
    )

def response_time_evaluator(run: Run, example: Example) -> EvaluationResult:
    """
    Оценивает время ответа системы
    """
    if not run.start_time or not run.end_time:
        return EvaluationResult(
            key="response_time",
            score=0.0,
            comment="Время выполнения не определено"
        )
    
    duration = (run.end_time - run.start_time).total_seconds()
    
    # Оценка времени ответа
    if duration < 5:
        score = 1.0
        comment = f"Отличное время ответа: {duration:.2f}с"
    elif duration < 15:
        score = 0.7
        comment = f"Хорошее время ответа: {duration:.2f}с"
    elif duration < 30:
        score = 0.4
        comment = f"Приемлемое время ответа: {duration:.2f}с"
    else:
        score = 0.1
        comment = f"Медленный ответ: {duration:.2f}с"
    
    return EvaluationResult(
        key="response_time",
        score=score,
        comment=comment
    )

class LocalGPTEvaluator:
    """Класс для оценки качества LocalGPT системы"""
    
    def __init__(self):
        self.client = get_langsmith_client()
        self.evaluators = [
            relevance_evaluator,
            completeness_evaluator,
            response_time_evaluator
        ]
    
    def create_test_dataset(self, test_queries: List[Dict[str, Any]]) -> Optional[str]:
        """
        Создает тестовый датасет в LangSmith
        """
        if not self.client:
            logger.warning("LangSmith не настроен, создание датасета пропущено")
            return None
        
        try:
            dataset_name = "LocalGPT_Test_Dataset"
            
            # Создаем датасет
            dataset = self.client.create_dataset(
                dataset_name=dataset_name,
                description="Тестовые запросы для оценки LocalGPT"
            )
            
            # Добавляем примеры
            examples = []
            for query_data in test_queries:
                example = self.client.create_example(
                    dataset_id=dataset.id,
                    inputs={"query": query_data["query"]},
                    outputs={"expected_content": query_data.get("expected", "")},
                    metadata=query_data.get("metadata", {})
                )
                examples.append(example)
            
            logger.info(f"Создан датасет {dataset_name} с {len(examples)} примерами")
            return dataset.id
            
        except Exception as e:
            logger.error(f"Ошибка создания датасета: {e}")
            return None
    
    def run_evaluation(self, dataset_name: str, experiment_name: str = None) -> Optional[Dict]:
        """
        Запускает оценку на датасете
        """
        if not self.client:
            logger.warning("LangSmith не настроен, оценка пропущена")
            return None
        
        try:
            def target_function(inputs: Dict) -> Dict:
                """Заглушка для целевой функции - в реальности здесь будет вызов LocalGPT"""
                # Здесь должен быть вызов вашей системы
                return {"content": "Заглушка ответа для тестирования"}
            
            # Запуск оценки
            results = evaluate(
                target_function,
                data=dataset_name,
                evaluators=self.evaluators,
                experiment_prefix=experiment_name or "LocalGPT_Evaluation"
            )
            
            logger.info(f"Оценка завершена: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка оценки: {e}")
            return None

# Предопределенные тестовые запросы
DEFAULT_TEST_QUERIES = [
    {
        "query": "Что такое hero_id и как он используется в логах?",
        "expected": "hero_id - идентификатор героя в игре",
        "metadata": {"category": "logs", "difficulty": "easy"}
    },
    {
        "query": "Проанализируй статистику матчей за последние дни",
        "expected": "Анализ должен включать количество матчей, винрейт, и статистику по героям",
        "metadata": {"category": "analytics", "difficulty": "medium"}
    },
    {
        "query": "Объясни механику получения опыта в игре",
        "expected": "Механика опыта должна быть объяснена на основе документации",
        "metadata": {"category": "documentation", "difficulty": "medium"}
    }
]

def setup_evaluation_suite():
    """Настройка полного набора для оценки"""
    if not is_tracing_enabled():
        print("⚠️  LangSmith не настроен. Установите LANGSMITH_API_KEY для использования оценок.")
        return None
    
    evaluator = LocalGPTEvaluator()
    dataset_id = evaluator.create_test_dataset(DEFAULT_TEST_QUERIES)
    
    if dataset_id:
        print(f"✅ Датасет создан: {dataset_id}")
        print("🔧 Для запуска оценки используйте: evaluator.run_evaluation('LocalGPT_Test_Dataset')")
    
    return evaluator

if __name__ == "__main__":
    # Пример использования
    setup_evaluation_suite()