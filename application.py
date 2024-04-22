import os
import PyPDF2
import typesense
import openai
from dotenv import load_dotenv

load_dotenv() ## load all the environment variables
api_key=os.getenv("OPENAI_API_KEY")
api_key=os.getenv("TYPESENSE_API_KEY")

def read_local_textbooks(folder_path):
    textbooks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                content = ""
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    content += page.extract_text()
            textbooks.append(content)
    return textbooks

folder_path = "Dataset"
textbooks_content = read_local_textbooks(folder_path)

from sentence_transformers import SentenceTransformer

def generate_embeddings(text):
    # Load a pre-trained model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Generate embeddings for the text
    embeddings = model.encode([text])
    return embeddings

# Example usage:
textbook_content = "This is the content of the textbook."
textbook_embeddings = generate_embeddings(textbook_content)

# Read the content of the textbook
with open("Dataset/fepw1ps.pdf", "rb") as file:
    textbook_content = file.read()

# Generate embeddings for the textbook content
textbook_embeddings = generate_embeddings(textbook_content)



client = typesense.Client({
    'nodes': [{
    'host': 'j1sfky02wgqzev79p-1.a1.typesense.net',  
    'port': '443',      
    'protocol': 'https'
    }],
    'api_key': 'TYPESENSE_API_KEY'       
})


collection_schema = {
    'name': 'textbook_embeddings',
    'fields': [
        {'name': 'text', 'type': 'string'},
        {'name': 'embeddings', 'type': 'vector'}
    ],
    'default_sorting_field': 'text'
}

client.collections.create(collection_schema)


for i, embedding in enumerate(textbook_embeddings):
    document = {
        'text': f'page_{i+1}', 
        'embeddings': embedding
    }
    client.collections['textbook_embeddings'].documents.create(document)

def generate_response(question, context):
    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=f"Q: {question}\nContext: {context}\nA:",
        max_tokens=50
    )
    return response.choices[0].text.strip()

from chainlit import Chainlit

# Initialize Chainlit instance
chainlit = Chainlit()

# Define conversation flow
@chainlit.step("user")
def user_step():
    question = input("You: ")
    return question

@chainlit.step("AI")
def ai_step(question):
    response = generate_response(question, textbook_content)
    print("AI:", response)
    return response

# Run the conversation loop
chainlit.run()