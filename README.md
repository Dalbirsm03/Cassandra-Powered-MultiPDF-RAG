# Cassandra-Powered-MultiPDF-RAG-with-Gemini-AI
Gemini PDF Chatbot is a web application that enables users to upload PDF documents, which are then processed and indexed. Using Google Generative AI (Gemini), users can interact with the PDF content by asking questions and receiving contextual responses based on the document's content.

The application utilizes Cassandra for storing PDF data, and leverages LangChain for text chunking and similarity search. The combination of these technologies allows for efficient querying and generating intelligent responses from the uploaded PDFs.

## Key Features:
- **PDF Text Extraction:** Extracts text from uploaded PDF files.
- **Text Chunking:** Splits large text into smaller, manageable chunks for better processing.
- **Vector Search:** Uses Cassandra's ANN (Approximate Nearest Neighbor) search for fast similarity matching.
- **Generative Responses:** Uses Google Gemini (Gemini-1.5-pro) to refine and generate intelligent responses based on the extracted content.
- **Interactive Chat Interface:** Users can ask questions about the uploaded PDFs and get detailed answers.

## Technologies Used:
- **Google Generative AI (Gemini):** For generating refined answers and understanding the context of the text.
- **Cassandra:** A NoSQL database for storing and indexing the vectorized content from PDFs.
- **LangChain:** For splitting large text into manageable chunks and managing the interaction between models.
- **Streamlit:** For creating an interactive web interface for users to upload PDFs and interact with the chatbot.
