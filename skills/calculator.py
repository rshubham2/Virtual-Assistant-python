import re
import wolframalpha

# Replace 'YOUR_APP_ID' with your actual Wolfram Alpha App ID
client = wolframalpha.Client('4GPA8Y-6KYJ26PL6A')


def calculate(expression):
    try:
        res = client.query(expression)
        result = next(res.results).text
        return f"The result of '{expression}' is {result}"
    except:
        return "Sorry, I couldn't calculate that. Please make sure your expression is valid."


def calculator_skill(text):
    math_expression = re.search(r'calculate\s+(.*)', text, re.IGNORECASE)
    if math_expression:
        return calculate(math_expression.group(1))
    else:
        return "To use the calculator, say something like 'Calculate the derivative of x^2'"
