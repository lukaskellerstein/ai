from huggingface_hub import snapshot_download

model_id = "lmsys/vicuna-13b-v1.5"
snapshot_download(
    repo_id=model_id,
    local_dir="vicuna-hf",
    local_dir_use_symlinks=False,
    revision="main",
)
