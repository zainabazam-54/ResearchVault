ResearchVault --- Private Research Paper Q&A
<img width="1920" height="826" alt="Screenshot (962)" src="https://github.com/user-attachments/assets/7c090565-86bf-42bc-9524-a86bc506ced6" />

ResearchVault is a retrieval-augmented generation (RAG) chatbot designed
to let users ask questions about a collection of research papers and
receive context-grounded answers.

This project demonstrates a dense retrieval workflow using Hugging Face
sentence embeddings, FAISS vector search, LangChain, Google Gemini, and
a Gradio chat interface.

Privacy note: The research papers used during development were
accessed privately through university resources. They are not included
in this public repository. The vector database and extracted document
text are also intentionally excluded. To run the project, use
documents you have permission to process and share.

Features
- Load PDF research papers and extract their text
- Split documents into overlapping text chunks
- Generate dense vector embeddings using
  sentence-transformers/all-MiniLM-L6-v2
- Index embeddings in a FAISS vector store
- Retrieve relevant chunks using similarity search
- Generate context-based answers using Google Gemini
- Chat through a Gradio interface
  
