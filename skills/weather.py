import requests


def get_weather(city):
    # Replace 'your_api_key_here' with a real OpenWeatherMap API key
    api_key = "5349842d199b7f906f6c87e50ab127e8"
    base_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"

    response = requests.get(base_url)
    if response.status_code == 200:
        data = response.json()
        temp = data['main']['temp'] - 273.15  # Convert from Kelvin to Celsius
        description = data['weather'][0]['description']
        return f"The weather in {city} is {description} with a temperature of {temp:.1f}°C"
    else:
        return "Sorry, I couldn't fetch the weather information at the moment."


def weather_skill(text):
    # Extract city name from the text (this is a simple implementation)
    words = text.split()
    if "in" in words:
        city = words[words.index("in") + 1]
    else:
        return "Please specify a city. For example, 'What's the weather in London?'"

    return get_weather(city)