import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


DB_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore")

# Initialize embeddings (same model as when saving)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the existing FAISS vector store
db = FAISS.load_local(
    DB_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)


print("Vector DB loaded")


# -----------------------------
# Gemini Client
# -----------------------------

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


model = genai.GenerativeModel('gemini-3-flash-preview')



# -----------------------------
# RAG Chat Function
# -----------------------------
def bot_response(user_message, history):
    # Retrieve documents
    docs = db.similarity_search(user_message, k=5)
    context = "\n\n".join([doc.page_content for doc in docs])

    
    prompt = f"Context:\n{context}\n\nUser Question: {user_message}\n\n" \
             "Instruction: Answer ONLY using the context. Use academic tone and bullet points."

    # Gemini call
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.2, "max_output_tokens": 1200}
    )

    bot_message = ""
    if response.candidates:
            for part in response.candidates[0].content.parts:
                # Skip thought parts and only take the final answer text
                if hasattr(part, "text") and not getattr(part, "thought", False):
                    bot_message += part.text

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": bot_message})
    return history

with gr.Blocks() as demo:
    gr.Markdown("# ResearchVault Private Document Chatbot")
    chatbot = gr.Chatbot(height=450)  # remove type="messages"
    user_input = gr.Textbox(placeholder="Ask a question...", show_label=False)

    send_btn = gr.Button("Send")
    clear_btn = gr.Button("Clear")

    send_btn.click(
        bot_response,
        inputs=[user_input, chatbot],
        outputs=chatbot
    )

    user_input.submit(
        bot_response,
        inputs=[user_input, chatbot],
        outputs=chatbot
    )

    clear_btn.click(lambda: [], outputs=chatbot)

demo.launch()
