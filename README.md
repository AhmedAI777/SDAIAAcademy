# SDAIAAcademy
Problem:

University information is often distributed across multiple PDF documents, such as course catalogs, academic regulations, student handbooks, and program information. Finding a specific piece of information requires users to manually search through long and sometimes complex documents.
Traditional keyword-based searching may also fail when the user's question uses different wording from the information contained in the documents. This can make it difficult for students and university staff to quickly find accurate and relevant information.
The project addresses this problem by providing a question-answering system that can retrieve relevant information from university documents and present it in a simple interface.

Solution:
The University Knowledge Assistant is a Retrieval-Augmented Generation (RAG) application that allows users to upload university PDF documents and ask questions about their content.
The system processes the documents by extracting their text, dividing the text into manageable chunks, generating vector embeddings, and storing them in ChromaDB. When a user submits a question, the system generates an embedding for the query and uses ChromaDB to retrieve the most relevant document chunks using cosine similarity.
The retrieved information is then provided to an LLM through OpenRouter to generate a grounded answer based only on the available university documentation.


The application also provides retrieval evaluation information, including:
Chunks Indexed
Chunks Retrieved
Best Distance
Average Distance
Best Cosine Similarity
Average Cosine Similarity
Response Time
Sources Included

The Engineer has been requested to build a Full RAG Application. which means to select a problem to solve it by AI models. 
Modren Data Engineering for Advanced Al Systems, [SDAIA Academy]
