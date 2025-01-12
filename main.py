import subprocess
import os
import PyPDF2
from diffusers import DiffusionPipeline
from PIL import Image
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
# Define the system prompts for Regular and RAG modes with instructions for detailed responses
regular_prompt = (
    "Your name is CloneGPT and You are a helpful assistant working on social robotics. Please provide detailed, "
    "comprehensive responses to the user's queries based solely on the conversation history."
)

rag_prompt = (
    "You are an AI assistant with access to specific documents. Please provide thorough "
    "answers based on the document content provided, covering all relevant points in detail."
)

# Separate conversation histories for Regular and RAG modes
regular_conversation_history = regular_prompt
rag_conversation_history = rag_prompt

# Paths for the model and mode
MODEL_NAME = "gemma:2b"


# Function to interact with the LLM using the Ollama CLI
def chat_with_llm(prompt, model=MODEL_NAME):
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            stdout=subprocess.PIPE,  # Capture stdout
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
            text=True,
            encoding='utf-8'
        )
        output = result.stdout.strip()
        # Filter out specific unwanted messages
        filtered_output = '\n'.join(
            line for line in output.splitlines()
            if "failed to get console mode" not in line
        )
        if result.returncode != 0:
            return f"Error: {filtered_output}"
        return filtered_output
    except Exception as e:
        return f"An error occurred: {str(e)}"


# Function to load and extract text from a PDF using PyPDF2
def load_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in range(len(reader.pages)):
                text += reader.pages[page].extract_text()
            return text
    except Exception as e:
        return f"An error occurred while reading the PDF: {str(e)}"


# Function to handle RAG by combining the query with document content
def chat_with_rag(query, documents, model=MODEL_NAME):
    global rag_conversation_history
    combined_input = rag_conversation_history + "\nUser Query: " + query + "\nDocument Content: " + documents
    response = chat_with_llm(combined_input, model=model)
    rag_conversation_history += f"\nUser: {query}\nLLM: {response}"
    return response


# Function to handle Regular conversation
def chat_with_regular(query, model=MODEL_NAME):
    global regular_conversation_history
    combined_input = regular_conversation_history + "\nUser Query: " + query
    response = chat_with_llm(combined_input, model=model)
    regular_conversation_history += f"\nUser: {query}\nLLM: {response}"
    return response


if __name__ == "__main__":
    mode = "regular"
    print("Starting LLM Chat. Type 'text to image' to generate image, 'switch to RAG' to switch to RAG mode and 'switch to regular' to switch back.")

    while True:
        user_input = input("Enter command or query: ")

        if user_input == "switch to RAG":
            mode = "RAG"
            print("Switched to RAG mode")
        elif user_input == "switch to regular":
            mode = "regular"
            print("Switched to regular mode")
        elif user_input == "text to image":
            mode = "SD"
            print("Switched to Text-to-Image")
        elif mode == "regular":
            response = chat_with_regular(user_input)
            print(f"LLM (Regular): {response}")
        elif mode == "SD":
            image = pipe(user_input).images[0]
            image_path = "output_image.png"
            image.save(image_path)
        elif mode == "RAG":
            pdf_content = load_pdf("sample.pdf")  # Replace with your actual PDF path
            if "An error occurred while reading the PDF" in pdf_content:
                print(pdf_content)
            else:
                response = chat_with_rag(user_input, pdf_content)
                print(f"LLM (RAG): {response}")
