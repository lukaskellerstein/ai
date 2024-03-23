from huggingface_hub import HfApi
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv())  # read local .env file

API_TOKEN = os.environ.get("HF_TOKEN")
print(API_TOKEN)

api = HfApi(token=API_TOKEN)

model_id = "lukaskellerstein/mistral-7b-lex-4bit-v1.0"

# Create a new repository for the model
api.create_repo(model_id, exist_ok=True, repo_type="model")

# Upload the model and tokenizer
api.upload_folder(
    repo_id=model_id,
    folder_path="/home/lukas/Models/3_example-qlora/SAVED_FINE-TUNED/ALL",
    path_in_repo="",
)
