import os
from PIL import Image
from google import genai
from google.genai.errors import APIError

from dotenv import load_dotenv # Make sure python-dotenv is installed!

# This line loads variables from the local .env file into the environment
load_dotenv() 


# --- Setup ---
# 1. Get an API key from Google AI Studio and set it as an environment variable 
#    named GEMINI_API_KEY. The client will automatically pick it up.
#    You can also pass it directly: client = genai.Client(api_key="YOUR_API_KEY")

try:
    # Initialize the Gemini client
    client = genai.Client()
    print("✅ Gemini client initialized successfully!")
except Exception as e:
    print(f"❌ Error initializing Gemini client: {e}")
    print("Please ensure you have set the GEMINI_API_KEY environment variable.")
    exit()

def generate_caption_gemini(image_path, prompt="Describe this image in detail"):
    """
    Generates a caption for an image using the Gemini Pro Vision model.
    
    Args:
        image_path (str): The file path to the image.
        prompt (str): The instruction for the model.
        
    Returns:
        str: The generated caption or an error message.
    """
    print(f"📸 Processing image: {image_path}")
    
    try:
        # Load the image using PIL
        img = Image.open(image_path).convert("RGB")
        
        # The content list for the model takes the prompt and the image
        content = [prompt, img]
        
        # Call the Gemini Pro Vision model
        # 'gemini-2.5-flash' is excellent for multimodal tasks like this.
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=content
        )
        
        return response.text
        
    except FileNotFoundError:
        return f"❌ Error: Image file not found at {image_path}"
    except APIError as e:
        return f"❌ Gemini API Error: {e}"
    except Exception as e:
        return f"❌ An unexpected error occurred: {e}"

if __name__ == "__main__":
    # Ensure the Google GenAI library is installed: pip install google-genai
    
    image_path = input("🖼️ Enter image path: ").strip()
    prompt = "Analyze this EV charging network user interface. For each 'Kiosk' section (Kiosk 1, Kiosk 2, Kiosk 3), describe its status, any active promotions, and specifically describe the visual content of any images or graphics displayed within its promotion or coupon section. Use a structured format for each kiosk. And if if any kiosk's status is occupied then do not need to give this much information just say its occupied and try to mainly focus on the image shown as coupon in the other available kiosks. If there are no images or graphics displayed within a kiosk's promotion or coupon section, simply state 'No visual content available"
    
    # Use a more descriptive default prompt for better results from Gemini
    if not prompt:
        prompt = "Describe this image in detail"
    
    result = generate_caption_gemini(image_path, prompt)
    print("\n🧠 Model output:\n", result)