import requests

# Функция для интеграции в LLM через tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the weather for, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
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
    }
]


def get_current_weather(location: str) -> str:
    """
    Get the current weather for a location.

    Args:
        location (str): The location to get the weather for, e.g. San Francisco, CA.

    Returns:
        str: The weather in a certain city.

    Raises:
        Exception: An error occurred due to server unavailability or incorrect parameters were passed.
    """
    api_key: str = "13acbc70131b46fb940125713242410"
    url: str = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}&lang=ru"
    response = requests.get(url, timeout=60)
    if response.status_code != 200:
        raise Exception(response.text)
    weather_data: dict = response.json()
    return f'На данный момент сейчас температура {weather_data["current"]["temp_c"]} градусов по Цельсию. ' \
           f'Погода - { weather_data["current"]["condition"]["text"]}. Локация - {weather_data["location"]["name"]}'


def calculate(operation: str, number_one: int, number_two: int) -> str:
    """
    Performs a mathematical operation (addition, subtraction, multiplication, division) on two numbers.

    Args:
        operation (str): The mathematical operation to perform. Supported operations: add, subtract, multiply, divide.
        number_one (int): The first integer number.
        number_two (int): The second integer number.

    Returns:
        str: An operation performed with two numbers.

    Raises:
        ValueError: An error occurred with an unknown operation.
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


if __name__ == "__main__":
    get_current_weather("London")
