import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("API_KEY"),
)

conversation = []
while True:
    user_message = input("Enter your message: ")
    if user_message.lower() == "exit":
        break

    conversation.append({"role": "user", "content": user_message})
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=conversation
    )

    bot_message = message.content[0].text
    print(f"Claude: {bot_message}")
    conversation.append({"role": "assistant", "content": bot_message})
