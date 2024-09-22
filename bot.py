import os
import gradio as gr
from google.cloud import vision
import google.generativeai as genai

# Configure the API keys for Generative AI and Vision API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

genai.configure(api_key=GOOGLE_API_KEY)

# Set up the Vision API client
client = vision.ImageAnnotatorClient()

# Initialize the Generative AI model
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
    
    description = ', '.join([label.description for label in labels])
    
    return f"This image contains: {description}" if description else "No descriptive information could be found."

def chatbot(prompt, image=None):
    if image:
        # Describe image if one is provided
        return describe_image(image), None  # Clear the prompt input
    else:
        # Generate text-based response if no image is provided
        response = chat.send_message(prompt, stream=True)
        generated_response = ""
        for chunk in response:
            if chunk.text:
                generated_response += chunk.text
        return generated_response, None  # Clear the prompt input after response

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# AI Chatbot with Image Analysis")
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(label="Ask me anything", placeholder="Type your question here...", lines=2)
            image_input = gr.Image(label="Upload an image", type="filepath")
        
        with gr.Column():
            output = gr.Textbox(label="Response", lines=6)
    
    submit_button = gr.Button("Submit")

    # Allow both submit button click and Enter key to trigger the chatbot
    prompt_input.submit(fn=chatbot, inputs=[prompt_input, image_input], outputs=[output, prompt_input])
    submit_button.click(fn=chatbot, inputs=[prompt_input, image_input], outputs=[output, prompt_input])

# Launch the Gradio app with shareable link
demo.launch(share=True)
