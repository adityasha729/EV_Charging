import os
from PIL import Image
from google import genai
from google.genai.errors import APIError
from dotenv import load_dotenv

load_dotenv()

try:
    client = genai.Client()
except Exception as e:
    client = None
    print(f"❌ Error initializing Gemini client: {e}")

def generate_caption_gemini(image_path, prompt=None):
    """
    Generates a caption for an image using the Gemini Pro Vision model.
    Args:
        image_path (str): The file path to the image.
        prompt (str): The instruction for the model.
    Returns:
        str: The generated caption or an error message.
    """
    if client is None:
        return "❌ Gemini client not initialized."
    if not prompt:
        prompt = "Describe this image in detail"
    try:
        img = Image.open(image_path).convert("RGB")
        content = [prompt, img]
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=content
        )
        return response.text
    except FileNotFoundError:
        return f"❌ Error: Image file not found at {image_path}"
    except APIError as e:
        return f"❌ Gemini API Error: {e}"
    except Exception as e:
        return f"❌ An unexpected error occurred: {e}"
