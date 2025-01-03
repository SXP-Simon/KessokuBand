import copy
import io
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import Dataset
import transformers
import openmind
from peft import LoraConfig, get_peft_model, TaskType

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

PROMPT_DICT = {
    "prompt_input": (
        "Instruction: {instruction}\n"
        "Question: {question}\n"
        "Options:\n"
        "(A) {option_A}\n"
        "(B) {option_B}\n"
        "(C) {option_C}\n"
        "(D) {option_D}\n"
        "Response: The correct answer is ({output})."
    ),
    "prompt_no_input": (
        "Instruction: {instruction}\n"
        "Response: {output}"
    ),
}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(openmind.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    output_dir: str = field(default="./output/Qwen1_5_peft", metadata={"help": "Output directory for model and logs."})

def smart_tokenizer_and_embedding_resize(special_tokens_dict: Dict, tokenizer, model):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def _tokenize_fn(strings: Sequence[str], tokenizer) -> Dict:
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(input_ids=input_ids, input_ids_lens=input_ids_lens)

def jload(f, mode="r"):
    if not isinstance(f, io.IOBase):
        with open(f, mode=mode) as f:
            jdict = json.load(f)
    else:
        jdict = json.load(f)
    return jdict

def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer) -> Dict:
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = [input_id.clone() for input_id in input_ids]
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer):
        super(SupervisedDataset, self).__init__()
        list_data_dict = jload(data_path)

        sources = []
        for example in list_data_dict:
            if len(example["option"]) != 4:
                raise ValueError("Each example must have exactly 4 options.")
            source = PROMPT_DICT["prompt_input"].format_map({
                "instruction": example["instruction"],
                "question": example["question"],
                "option_A": example["option"][0],
                "option_B": example["option"][1],
                "option_C": example["option"][2],
                "option_D": example["option"][3],
                "output": example["output"]
            })
            sources.append(source)

        targets = [f"The correct answer is ({example['output']}).{tokenizer.eos_token}" for example in list_data_dict]

        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: object

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer, data_args) -> Dict:
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = openmind.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)

    tokenizer = openmind.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = openmind.Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()

    trainer.save_model(output_dir=training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    model.save_pretrained(training_args.output_dir)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(f"{training_args.output_dir}/checkpoint-200")

    logging.info(f"Model saved to {training_args.output_dir}")

if __name__ == "__main__":
    train()