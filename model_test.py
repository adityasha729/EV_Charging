
import os
from dotenv import load_dotenv
from langchain.llms import OpenAI, GooglePalm

def test_model(prompt, bot_name="gemini"):
    """
    Test the chatbot models with a given prompt.
    bot_name: 'gemini' or 'openai'
    Returns the model's output as a string.
    """
    chatbots = {
        "gemini": {
            "api_key": os.environ.get("GEMINI_API_KEY"),
            "model": os.environ.get("GEMINI_MODEL")
        },
        # "openai": {
        #     "api_key": os.environ.get("OPENAI_API_KEY"),
        #     "model": os.environ.get("OPENAI_MODEL")
        # }
    }
    if bot_name not in chatbots:
        raise ValueError(f"Unknown bot: {bot_name}")
    if bot_name == "gemini":
        llm = GooglePalm(google_api_key=chatbots["gemini"]["api_key"], model_name=chatbots["gemini"]["model"])
    else:
        llm = OpenAI(openai_api_key=chatbots["openai"]["api_key"], model_name=chatbots["openai"]["model"])
    return llm(prompt)

if __name__ == "__main__":
    load_dotenv()
    print("Model Test Utility")
    print("Type 'exit' to quit.")
    bot = input("Choose model (gemini/openai) [gemini]: ").strip().lower() or "gemini"
    while True:
        prompt = input("Prompt: ")
        if prompt.lower() == "exit":
            break
        try:
            output = test_model(prompt, bot)
            print(f"\n[{bot} output]:\n{output}\n")
        except Exception as e:
            print(f"Error: {e}")
