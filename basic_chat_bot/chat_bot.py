from projects.chat_bot.utils import extract_reply, process_tool_call 
import anthropic
import json
from projects.chat_bot.database import FakeDatabase
from pathlib import Path
import os
from dotenv import load_dotenv

def simple_chat():
    # initisalize the anthropic client and the database instance
    config_path = Path(__file__).parent / "parameters.json"
    config = json.load(open(config_path))
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)   # reads .env and populates os.environ
    client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))
    db_instance = FakeDatabase()

    user_message = input("\nUser: ")
    messages = [{"role": "user", "content": user_message}]
    
    while user_message.lower() not in ["exit", "quit", "bye", "goodbye", "q"]:
        #If the last message is from the assistant, get another input from the user
        if messages[-1].get("role") == "assistant":
            user_message = input("\nUser: ")
            messages.append({"role": "user", "content": user_message})

        #Send a request to Claude
        response = client.messages.create(
            model=config["model_name"],
            system="\n".join(config["system"]),
            max_tokens=config["max_tokens"],
            tools=config["tools"],
            messages=messages,
        )
        # Update messages to include Claude's response
        messages.append(
            {"role": "assistant", "content": response.content}
        )

        if response.stop_reason == "tool_use":
            # tool_use = response.content[-1] #Naive approach assumes only 1 tool is called at a time
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            for tool in tool_uses:  # handles 1 or many
                tool_result = process_tool_call(tool.name, tool.input, db_instance)

                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool.id,
                                "content": str(tool_result),
                            }
                        ],
                    },
                ) 
        
        model_reply = "".join(extract_reply(b.text) for b in response.content if b.type == "text")
        if len(model_reply) > 0:
            print("\nTechNova Support: " + f"{model_reply}" )