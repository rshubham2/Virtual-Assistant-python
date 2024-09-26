todo_list = []

def add_todo(item):
    todo_list.append(item)
    return f"Added '{item}' to your to-do list."

def remove_todo(item):
    if item in todo_list:
        todo_list.remove(item)
        return f"Removed '{item}' from your to-do list."
    else:
        return f"'{item}' not found in your to-do list."

def show_todo():
    if todo_list:
        return "Your to-do list:\n" + "\n".join(f"- {item}" for item in todo_list)
    else:
        return "Your to-do list is empty."

def todo_skill(text):
    if "add to todo" in text.lower():
        item = text.lower().split("add to todo")[-1].strip()
        return add_todo(item)
    elif "remove from todo" in text.lower():
        item = text.lower().split("remove from todo")[-1].strip()
        return remove_todo(item)
    elif "show todo" in text.lower():
        return show_todo()
    else:
        return "To use the to-do list, say 'Add to todo [item]', 'Remove from todo [item]', or 'Show todo'"