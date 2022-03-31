import torch
import os

from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, RobertaForCausalLM, get_scheduler

from torch.utils.data import DataLoader
from torch.optim import AdamW

from tqdm.auto import tqdm

from datasets import load_metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def train_classifier(dataset):

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    def tokenize(examples):
        tokens = tokenizer(examples["premise_hypothesis"],
                           padding="max_length", max_length=512, truncation=True)
        tokens["labels"] = examples["label"]
        return tokens

    tokenized_dataset = dataset.map(
        tokenize, batched=True, load_from_cache_file=False, num_proc=4)
    tokenized_dataset.set_format(
        "torch", columns=['input_ids', 'attention_mask', 'labels'])

    train_dataloader = DataLoader(
        tokenized_dataset["train"].select(range(10000)), shuffle=True, batch_size=32)

    eval_dataloader = DataLoader(tokenized_dataset["test"].select(range(10000)), batch_size=32)

    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=3)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_epochs = 3

    num_training_steps = num_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))

    model.train()

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    metric = load_metric("accuracy")

    model.eval()

    for batch in eval_dataloader:

        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():

            outputs = model(**batch)

        logits = outputs.logits

        predictions = torch.argmax(logits, dim=-1)

        metric.add_batch(predictions=predictions, references=batch["labels"])

    metric.compute()

    __check_dir_exists("models")

    model.save_pretrained("./models/classifier")


def train_explanator(dataset):

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    classifier_model = RobertaForSequenceClassification.from_pretrained(
        "./models/classifier", num_labels=3)
    classifier_model.to(device)

    explanation_model = RobertaForCausalLM.from_pretrained(
        "roberta-base", is_decoder=True)
    explanation_model.to(device)

    def tokenize(examples):
        processed = {}
        tokens = tokenizer(examples["premise_hypothesis"], padding="max_length",
                           max_length=512, truncation=True, return_tensors="pt")

        processed["labels"] = tokenizer(examples["explanation_1"], padding="max_length",
                                        max_length=512, truncation=True, return_tensors="pt")["input_ids"]
        processed["encoder_hidden_states"] = classifier_model(
            **tokens, output_hidden_states=True)["hidden_states"][-1]
        return processed

    tokenized_dataset = dataset.map(
        tokenize, batched=True, load_from_cache_file=False, num_proc=4)
    tokenized_dataset.set_format("torch")

    train_dataloader = DataLoader(
        tokenized_dataset["train"].select(range(10000)), shuffle=True, batch_size=32)

    eval_dataloader = DataLoader(tokenized_dataset["test"].select(range(10000)), batch_size=32)

    optimizer = AdamW(explanation_model.parameters(), lr=5e-5)

    num_epochs = 3

    num_training_steps = num_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))

    explanation_model.train()

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = explanation_model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    metric = load_metric("accuracy")

    explanation_model.eval()

    for batch in eval_dataloader:

        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():

            outputs = explanation_model(**batch)

        logits = outputs.logits

        predictions = torch.argmax(logits, dim=-1)

        metric.add_batch(predictions=predictions, references=batch["labels"])

    metric.compute()

    __check_dir_exists("models")

    explanation_model.save_pretrained("./models/explanator")


def __check_dir_exists(dir: str):
    path = str(dir)
    if not os.path.exists(path):
        os.makedirs(path)
