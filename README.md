LLM-Based Folder Processor
This project is a tool that processes documents and files within a local folder, creates a knowledge base from them, and answers user queries using this knowledge base. The project is specifically designed to analyze PDF files and extract meaningful information from their content.

Project Purpose
The main goal of this project is to automatically analyze documents in a specified folder and store their content in a ChromaDB-based vector database. This allows users to easily access information within the documents using natural language queries.

Key Features
Document Scanning and Updating: Automatically detects and processes newly added or updated documents in the designated folder.

Efficient Processing: Saves system resources by only processing files that have changed.

Semantic Search: Enables semantic searches across documents using natural language queries.

LLM Integration: Provides meaningful and comprehensive answers to users by leveraging information from the processed documents.

Flexible and Extensible: Has an easily expandable architecture to support different document types (e.g., DOCX, TXT).

Installation
To install the project dependencies, use the following command:

pip install -r requirements.txt

Note: To create the requirements.txt file, you need to list all the Python libraries used in your project.

Usage
The application is started by running the backend/main.py file.

1. Specifying the Folder

Place the PDF files you want to process into the backend/documents folder.

2. Starting the Application

Run the following command in your terminal to start the application:

python backend/main.py

When the application starts, it will scan, process, and create a knowledge base from the PDF files in the backend/documents folder.

3. Submitting Queries

While the application is running, you can submit queries via the API or a command-line interface (CLI). For example:

# Example query code
from backend.main import ask_question

response = ask_question("What were the turning points of World War II?")
print(response)

Contributing
If you would like to contribute to the project, please open a pull request or report an issue. All contributions are welcome!

License: The project is released under the [License Name] license.

