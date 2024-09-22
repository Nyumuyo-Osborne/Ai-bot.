import os
from google.cloud import vision
import google.generativeai as genai
from google.cloud.vision_v1 import types

# Configure the API keys for Generative AI and Vision API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

genai.configure(api_key=GOOGLE_API_KEY)

# Set up the Vision API client
client = vision.ImageAnnotatorClient()

# List the models available and use one for text generation
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
        
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

def describe_image(image_path):
    """Function to describe an image using Google Cloud Vision API"""
    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.label_detection(image=image)
    labels = response.label_annotations
    
    # Extracting descriptions
    description = ', '.join([label.description for label in labels])
    
    if description:
        return f"This image contains: {description}"
    else:
        return "No descriptive information could be found."

while True:
    prompt = input("Ask me anything or type 'image' to analyze an image ('exit' to quit): ")
    if prompt.lower() == 'exit':
        break
    elif prompt.lower() == 'image':
        image_path = input("Enter the path of the image: ")
        if os.path.exists(image_path):
            image_description = describe_image(image_path)
            print(image_description)
        else:
            print("Invalid image path.")
    else:
        response = chat.send_message(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                print(chunk.text)
