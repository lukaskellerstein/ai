## 0_example-qlora

Codename: Joe podcast example

- custom dataset
- Trainer or SFTTrainer ????

TODO: Need to split each podcast to max 2048 tokens : https://medium.com/@geronimo7/from-transcripts-to-ai-chat-an-experiment-with-the-lex-fridman-podcast-3248d216ec16

https://medium.com/@geronimo7/from-transcripts-to-ai-chat-an-experiment-with-the-lex-fridman-podcast-3248d216ec16
https://medium.com/@geronimo7/finetuning-llama2-mistral-945f9c200611

## 1_example-qlora

Codename: None

https://blog.paperspace.com/mistral-7b-fine-tuning/

- mosaicml/instruct-v3 dataset
- SFTTrainer

## 2_example-qlora

[WORKS on Google Colab]

Codename: Guanaco

https://www.datacamp.com/tutorial/mistral-7b-tutorial

- guanaco-llama2-1k dataset
- SFTTrainer

## 3_example-qlora

[WORKS on Google Colab]

Codename: Lex Fridman podcast with dataset

- g-ronimo/lfpodcast dataset
- Trainer

https://medium.com/@geronimo7/finetuning-llama2-mistral-945f9c200611

## 4_example-qlora

Codename: ChatDoctor

- lavita/ChatDoctor-HealthCareMagic-100k dataset
- Trainer

https://github.com/sachink1729/Finetuning-Mistral-7B-Chat-Doctor-Huggingface-LoRA-PEFT/blob/main/mistral-finetuned%20(1).ipynb

https://sachinkhandewal.medium.com/finetuning-mistral-7b-into-a-medical-chat-doctor-using-huggingface-qlora-peft-5ce15d45f581

# TODO

0. use everywhere timedelta to measure time
1. Learn howto properly fine-tune a model
2. Learn howto run/chat with local model
3. Use DataCollator for Trainer instead of collation function
4. Choose between SFTTrainer and Trainer
5. User tensorFlow even locally ??
