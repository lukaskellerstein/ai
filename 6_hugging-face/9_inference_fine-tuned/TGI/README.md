docker run --gpus all --shm-size 1g -p 8080:80 -v $PWD/data:/data ghcr.io/huggingface/text-generation-inference:1.4 --model-id lukaskellerstein/mistral-7b-lex-4bit --quantize bitsandbytes-nf4

docker run --gpus all --shm-size 4g -p 8080:80 -v $PWD/data:/data ghcr.io/huggingface/text-generation-inference:1.4 --model-id lukaskellerstein/mistral-7b-lex-4bit-v1.0
