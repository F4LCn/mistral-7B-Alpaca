from datasets import load_from_disk
from transformers import AutoTokenizer
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datetime import datetime

if __name__ == '__main__':
    base_model_id = "mistralai/Mistral-7B-v0.1"
    model_max_length = 512
    project = "alpaca-finetune"
    base_model_name = "mistral"
    run_name = base_model_name + "-" + project
    output_dir = "./" + run_name
    dataset_name = "alpaca_code"
    dataset_dir = "./datasets/" + dataset_name

    packed_dataset = load_from_disk(dataset_dir)
    train_dataset = packed_dataset['train']
    eval_dataset = packed_dataset['eval']

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, model_max_length=model_max_length)
    tokenizer.pad_token = tokenizer.eos_token

    # base model
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb)

    # lora
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora)

    # trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=TrainingArguments(
            output_dir=output_dir,
            resume_from_checkpoint=f"{output_dir}/checkpoint-1000",
            warmup_steps=5,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=4,
            max_steps=2000,
            learning_rate=2.5e-5,
            bf16=True,
            optim="paged_adamw_8bit",
            logging_dir=f"./logs/{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
            logging_steps=10,
            save_steps=100,
            save_strategy="steps",
            eval_steps=200,
            evaluation_strategy="steps",
            do_eval=True,
            report_to=['tensorboard'],
            run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    model.config.use_cache = False

    # training loop
    trainer.train(resume_from_checkpoint=True)