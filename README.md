# Mistral 7B finetuning on alpaca (+ alpaca instruct & alpaca code)

This repo is my experimentation with finetuning the mistral 7B base model on alpaca.
I'm mainly using qlora and training on a single RTX 4090.

The model expects an alpaca style prompt (with optional input)

```
Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request

### Instruction:
{instruction}

[### Input:
{input}]

### Response:

```

The training data should be in the data folder (json array of ```{ 'instruction', 'input', 'output' }``` objects). The
dataset
prep notebook can be used to split/pack and pad the dataset and convert it into a huggingface dataset (saved to the
datasets folder).
Huggingface datasets can also be used directly if put in the datasets folder.

The base model (mistral 7B) is quantized to 4bit (nf4) and float16. Llama2-7B can technically be used as a dropdown
replacement for mistral-7B with almost no changes to code