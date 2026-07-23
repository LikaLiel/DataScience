from typing import Dict, Any
from projects.chat_bot.database import FakeDatabase
import re

def extract_reply(text):
    pattern = r'<reply>(.*?)</reply>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None

def process_tool_call(
    tool_name: str, 
    tool_input: Dict[Any, Any],
    db_instance: FakeDatabase,
):
    if tool_name == "get_user":
        return db_instance.get_user(tool_input["key"], tool_input["value"])
    elif tool_name == "get_order_by_id":
        return db_instance.get_order_by_id(tool_input["order_id"])
    elif tool_name == "get_customer_orders":
        return db_instance.get_customer_orders(tool_input["customer_id"])
    elif tool_name == "cancel_order":
        return db_instance.cancel_order(tool_input["order_id"])
    else:
        raise ValueError(f"Unsopported tool use: {tool_name}")