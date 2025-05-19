import gradio as gr
import os
from character_chatbot import CharacterChatbot
from character_chatbot.character_chatbotQwen import CharacterChatbotQwen
from text_classification import LocationClassifier
from dotenv import load_dotenv

load_dotenv()

def classify_text(text_classification_model, text_classification_data_path, text_to_classify):
    try:
        location_classifier = LocationClassifier(model_path=text_classification_model,
                                                data_path=text_classification_data_path,
                                                huggingface_token=os.getenv('huggingface_token'))
        output = location_classifier.classify_location(text_to_classify)
        return output
    except Exception as e:
        return f"Error: {str(e)}"
    
def chat_with_character(message, history):
    character_chatbot = CharacterChatbot(
        "christopherxzyx/StrangerThings_Llama-3-8B_v4",
        huggingface_token=os.getenv('huggingface_token'),
    )
    output = character_chatbot.chat(message,history)
    output = output['content'].strip()
    return output

def character_chatbot_withQwen(message, history):
    chatbotQwen = CharacterChatbotQwen(
        "christopherxzyx/StrangerThings_Qwen-3-4B",
        huggingface_token=os.getenv('huggingface_token'),
    )
    output = chatbotQwen.chat(message, history)
    output = output['content'].strip()
    return output

def main():
    with gr.Blocks() as interface:  
        # Text Classification (LLMs)
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Text Classification (LLMs)</h1>")
                with gr.Row():
                    with gr.Column():
                        text_classification_output = gr.Textbox(label="Text Classification Output")
                    with gr.Column():
                        text_classification_model = gr.Textbox(label="Model path")
                        text_classification_data_path = gr.Textbox(label="Data path")
                        text_to_classify = gr.Textbox(label="Text to classify")
                        classify_text_button = gr.Button("Classify Text (Location)")
                        classify_text_button.click(classify_text, inputs=[text_classification_model, text_classification_data_path, text_to_classify], outputs=[text_classification_output])             
        # Character Chatbot (LLMs)
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Chatbot (LLMs)</h1>")
                gr.ChatInterface(chat_with_character)
                
        #Character Chatbot Qwen 
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Chatbot (Qwen)</h1>")
                gr.ChatInterface(character_chatbot_withQwen)
                
    interface.launch(share=True, debug=True)  # Bật debug để xem log

if __name__ == "__main__":
    main()