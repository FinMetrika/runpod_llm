import sys, logging, random, json, warnings
from termcolor import colored
import pandas as pd
import numpy as np

import torch
from datasets import load_dataset
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer
)

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)

from config import ProjectConfig
import utils


# logging.basicConfig(filename="program_log.txt",
#                     level=logging.INFO,
#                     format='%(asctime)s - %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S')


#################################
def main():
    # Instantiate the config dataclass
    FLAGS = ProjectConfig()
    utils.update_config(FLAGS)
    #logging.info(f"Configuration: {FLAGS}")

    # Random seeds
    seed = FLAGS.seed
    random.seed(seed)  # python 
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # torch.cuda
    
    warnings.filterwarnings("ignore")

    # ----- OBJECTS -----
    DEVICE = utils.check_device()
    MODEL_NAME = FLAGS.model_name_hf
    INPUT_FILE = FLAGS.data_dir_path/FLAGS.input_file_name
    
    # ----- DATASET -----
    df = pd.read_csv(INPUT_FILE)
    dataset_data = [
        {
            "instruction": "Detect the sentiment of the tweet.",
            "input": row_dict["Tweet"],
            "output": row_dict["BERT Labels"]
        }
        for row_dict in df.to_dict(orient="records")
    ]

    with open("alpaca-bitcoin-sentiment-dataset.json", "w") as f:
        json.dump(dataset_data, f)

    # ----- LOAD MODEL -----
    print(colored(f'Loading model: {MODEL_NAME}', 'blue'))
    model = LlamaForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(colored(f'\nTOKENIZER ...', 'blue'))
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"

    data = load_dataset("json", data_files="alpaca-bitcoin-sentiment-dataset.json")

    def generate_prompt(data_point):
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501
    ### Instruction:
    {data_point["instruction"]}
    ### Input:
    {data_point["input"]}
    ### Response:
    {data_point["output"]}"""
    

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=FLAGS.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < FLAGS.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
    
        result["labels"] = result["input_ids"].copy()
    
        return result
    
    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    print(colored(f'\nTRAIN, VALID, TEST DATA ...', 'blue'))
    train_val = data["train"].train_test_split(
        test_size=200, shuffle=True, seed=42
    )
    train_data = (
        train_val["train"].map(generate_and_tokenize_prompt)
    )
    val_data = (
        train_val["test"].map(generate_and_tokenize_prompt)
    )

    # ----- TRAINING -----
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT= 0.05
    LORA_TARGET_MODULES = [
        "q_proj",
        "v_proj",
    ]
    
    BATCH_SIZE = 128
    MICRO_BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
    LEARNING_RATE = 3e-4
    TRAIN_STEPS = 10
    OUTPUT_DIR = "experiments"
    
    print(colored(f'\nLOAD PEFT MODEL ...', 'blue'))
    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    print(colored(f'\nTRAINING ARGUMENTS...', 'blue'))
    training_arguments = TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        max_steps=TRAIN_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=1,
        optim="adamw_torch",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=1,
        save_steps=1,
        output_dir=OUTPUT_DIR,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="tensorboard"
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    
    print(colored(f'\nTRAINING ...', 'blue'))
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_arguments,
        data_collator=data_collator
    )
    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))
    
    model = torch.compile(model)
    
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    
if __name__ == '__main__':
    #logging.info(f"Program started with arguments: {FLAGS}")
    main()