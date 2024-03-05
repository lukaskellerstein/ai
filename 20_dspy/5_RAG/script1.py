import os
import chromadb
from chromadb.utils import embedding_functions
# from dspy.retrieve import ChromadbRM
# import dspy

# Path to the folder containing .txt files
folder_path = "source"
# Collection name
collection_name = "my_collection"
# Initialize ChromaDB instance
embedding = embedding_functions.DefaultEmbeddingFunction()

# -------------------------------------
# -------------------------------------
# 1. Fill DB
# -------------------------------------
# -------------------------------------



# Function to load text files into ChromaDB
def load_txt_files_into_chromadb(folder_path, chromadb_instance, collection_name):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Filter out only .txt files
    txt_files = [f for f in files if f.endswith('.txt')]
    
    # Iterate through each .txt file and load into ChromaDB
    for txt_file in txt_files:
        file_path = os.path.join(folder_path, txt_file)
        with open(file_path, 'r') as file:
            # Read the content of the file
            content = file.read()
            # Load content into ChromaDB with specified collection name
            chromadb_instance.insert(content, collection=collection_name)

chromadb_instance = chromadb.ChromaDB(default_ef=embedding)

# Load .txt files into ChromaDB
load_txt_files_into_chromadb(folder_path, chromadb_instance, collection_name)

print("All .txt files loaded into ChromaDB successfully.")



# # -------------------------------------
# # -------------------------------------
# # 2. DSPy
# # -------------------------------------
# # -------------------------------------


# # model
# model = dspy.OllamaLocal(model='llama2')

# # db
# retriever_model = ChromadbRM(
#     collection_name,
#     folder_path,
#     embedding_function=embedding,
#     k=5
# )

# dspy.settings.configure(lm=model, rm=retriever_model)

# # -----------
# # Signature
# # -----------
# class GenerateAnswer(dspy.Signature):
#     """Answer questions with short factoid answers."""

#     context = dspy.InputField(desc="may contain relevant facts")
#     question = dspy.InputField()
#     answer = dspy.OutputField(desc="often between 1 and 5 words")


# # -----------
# # Pipeline
# # -----------
# class RAG(dspy.Module):
#     def __init__(self, num_passages=3):
#         super().__init__()

#         self.retrieve = dspy.Retrieve(k=num_passages)
#         self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
#     def forward(self, question):
#         context = self.retrieve(question).passages
#         prediction = self.generate_answer(context=context, question=question)
#         return dspy.Prediction(context=context, answer=prediction.answer)