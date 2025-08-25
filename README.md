LLM-Based Folder Processor
This project is a powerful tool that processes documents and files within a local folder, creating a knowledge base from them, and answering user queries using this knowledge base. It is specifically designed to analyze PDF files and extract meaningful information from their content.

Project Purpose
The main goal of this project is to automatically analyze documents in a specified folder and store their content in a ChromaDB-based vector database. This allows users to easily access information within the documents using natural language queries and receive meaningful responses.

Key Features
Intelligent Document Processing: Automatically scans and processes documents in the designated folder. This ensures that newly added or updated documents are instantly included in the knowledge base.

Efficient Update Mechanism: Processes only changed or newly added files to use resources most efficiently. This guarantees fast and effective operation even with large document collections.

Semantic Search Capability: Goes beyond keyword-based search by understanding the meaning within the documents and providing the most relevant results for user queries.

Advanced LLM Integration: Leverages information obtained from the processed documents to provide comprehensive and meaningful answers to users through a large language model (LLM) integration.

Flexible and Extensible Architecture: The project has a modular architecture that can be easily expanded to support different document types (e.g., DOCX, TXT).

Multi-Modal Content Support (Future Plan): There are future plans to add the ability to process different content types such as images and audio files.

Installation
Prerequisites

To run this project, you need to have the following components installed on your system:

Python 3.8 or above: You can check your version by running python --version.

pip: The Python package manager.

Steps

Clone the Repository: Clone the project to your local machine by running the following command in your terminal.

git clone https://github.com/berkay-uras/my-llm-folder-processor.git

Navigate to the Project Directory:

cd my-llm-folder-processor

Install Dependencies: Install all project dependencies using the requirements.txt file.

pip install -r requirements.txt

Note: If you don't have a requirements.txt file, you need to manually add the libraries you used in your project to this file.

Usage
Step 1: Specifying the Document Folder

Place the PDF files you want to process into the backend/documents folder.

Step 2: Starting the Application

Run the following command in your terminal to start the application:

python backend/main.py

When the application starts, it will scan, process, and create a knowledge base from the PDF files in the backend/documents folder.

Step 3: Submitting Queries

While the application is running, you can submit your queries via the API or a command-line interface (CLI).

Example query code
from backend.main import ask_question

response = ask_question("What were the turning points of World War II?")
print(response)

 In a development environment, you can create an API endpoint
 to submit queries via an HTTP POST request.
 This provides a more flexible interface.
 For example, with a RESTful API endpoint:
 POST /api/query {"question": "Your question here"}

Contributing
If you would like to contribute to the project, please feel free to open a pull request or report an issue. All contributions are welcome!
