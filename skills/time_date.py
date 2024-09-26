from datetime import datetime

def get_time():
    current_time = datetime.now().strftime("%I:%M %p")
    return f"The current time is {current_time}"

def get_date():
    current_date = datetime.now().strftime("%B %d, %Y")
    return f"Today's date is {current_date}"

def time_date_skill(text):
    if "time" in text.lower():
        return get_time()
    elif "date" in text.lower():
        return get_date()
    else:
        return "I can tell you the current time or date. Just ask!"