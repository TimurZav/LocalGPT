import requests
from __init__ import DATABASE_URL
from sqlalchemy import create_engine, text

# Функция для интеграции в LLM через tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given latitude and longitude",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "The latitude of a place",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "The longitude of a place",
                    },
                },
                "required": ["latitude", "longitude"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Performs a mathematical operation (addition, subtraction, multiplication, division) "
                           "on two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The mathematical operation to perform. "
                                       "Supported operations: add, subtract, multiply, divide",
                    },
                    "number_one": {
                        "type": "number",
                        "description": "The first number for the operation",
                    },
                    "number_two": {
                        "type": "number",
                        "description": "The second number for the operation",
                    },
                },
                "required": ["operation", "number_one", "number_two"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_sql_query",
            "description": "Generates an SQL query to retrieve data from a table with optional filtering, sorting "
                           "and limit",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "The name of the table from which to retrieve the data"
                    },
                    "columns": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "The list of columns to be selected from the table. If not specified, "
                                       "all columns (*) will be selected"
                    },
                    "conditions": {
                        "type": "string",
                        "description": "Optional conditions to filter the data (e.g., 'age > 30')"
                    },
                    "order_by": {
                        "type": "string",
                        "description": "Optional ordering for the data (e.g., 'age DESC')"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Optional limit on the number of rows returned"
                    }
                },
                "required": ["table_name"]
            }
        }
    }
]


def get_current_weather(latitude, longitude):
    """
    Используем OpenWeatherMap API или другой API для получения текущей погоды
    :param latitude:
    :param longitude:
    :return:
    """
    api_key = "13acbc70131b46fb940125713242410"
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={latitude},{longitude}&lang=ru"
    response = requests.get(url, timeout=60)
    if response.status_code != 200:
        return {"error": "Unable to fetch weather data"}
    weather_data = response.json()
    return f'На данный момент сейчас температура {weather_data["current"]["temp_c"]} градусов по Цельсию. ' \
           f'Погода - { weather_data["current"]["condition"]["text"]}. Локация - {weather_data["location"]["name"]}'


def calculate(operation, number_one, number_two):
    """
    Calculate digits.
    :param operation:
    :param number_one:
    :param number_two:
    :return:
    """
    if operation == "add":
        return f"Ответ является {number_one + number_two}"
    elif operation == "subtract":
        return f"Ответ является {number_one - number_two}"
    elif operation == "multiply":
        return f"Ответ является {number_one * number_two}"
    elif operation == "divide":
        return f"Ответ является {number_one / number_two}"
    else:
        raise ValueError(f"Unknown operation {operation}")


def generate_sql_query(table_name, columns="*", conditions=None, order_by=None, limit=None):
    """
    Генерирует SQL-запрос для получения данных из таблицы.

    :param table_name: Название таблицы, из которой нужно получить данные.
    :param columns: Список колонок для выбора (по умолчанию все колонки '*').
    :param conditions: Условия фильтрации в виде строки (например, 'age > 30').
    :param order_by: Условие сортировки (например, 'age DESC').
    :param limit: Лимит строк для выборки.
    :return: Сгенерированный SQL-запрос в виде строки.
    """
    # Если передан список колонок, преобразуем его в строку
    if isinstance(columns, list):
        columns = ', '.join(columns)

    # Начальный запрос
    query = f"SELECT {columns} FROM {table_name}"

    # Добавляем условия фильтрации, если они есть
    if conditions:
        query += f" WHERE {conditions}"

    # Добавляем условие сортировки, если оно есть
    if order_by:
        query += f" ORDER BY {order_by}"

    # Добавляем ограничение по количеству строк, если оно есть
    if limit:
        query += f" LIMIT {limit}"

    return get_data_from_sql_query(query)


def format_rows_as_string(rows, columns):
    """
    Форматирует данные строк и столбцов в виде строки, где сначала идут названия столбцов,
    а затем строки данных, разделенные запятыми.

    :param rows: Список строк данных.
    :param columns: Список названий столбцов.
    :return: Строка, представляющая данные в формате CSV.
    """
    # Формируем строку с названиями столбцов
    result = ', '.join(columns) + '\n'

    # Формируем строки с данными
    for row in rows:
        # Преобразуем каждую строку в строку, где значения разделены запятыми
        result += ', '.join(map(str, row)) + '\n'

    return result


def get_data_from_sql_query(query):
    """

    :param query:
    :return:
    """
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        try:
            result = conn.execute(text(query))
            columns = list(result.keys())
            rows = result.fetchall()
        except Exception as ex:
            print(ex)
            rows = []
    return format_rows_as_string(rows, columns)


if __name__ == "__main__":
    get_current_weather(45.0328, 38.9769)
