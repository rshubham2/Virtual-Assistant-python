# main.py
from assistant.core import VirtualAssistant
from skills.weather import weather_skill
from skills.time_date import time_date_skill
from skills.calculator import calculator_skill
from skills.todo_list import todo_skill
import time


def main():
    assistant = VirtualAssistant()

    # Add skills
    assistant.add_skill("weather", weather_skill)
    assistant.add_skill("time", time_date_skill)
    assistant.add_skill("date", time_date_skill)
    assistant.add_skill("calculate", calculator_skill)
    assistant.add_skill("todo", todo_skill)

    assistant.run()


if __name__ == "__main__":
    main()
