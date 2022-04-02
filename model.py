import torch
import os

from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, RobertaForCausalLM, get_scheduler

from torch.utils.data import DataLoader
from torch.optim import AdamW

from tqdm.auto import tqdm

from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def train_classifier(dataset):

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    def tokenize(examples):
        tokens = tokenizer(examples["premise_hypothesis"],
                           padding="max_length", max_length=250, truncation=True)
        tokens["labels"] = examples["label"]
        return tokens

    tokenized_dataset = dataset["train"].map(
        tokenize, batched=True, load_from_cache_file=False, num_proc=4, remove_columns=dataset["train"].column_names)
    tokenized_dataset.set_format(
        "torch", columns=['input_ids', 'attention_mask', 'labels'])

    train_dataloader = DataLoader(
        tokenized_dataset, shuffle=True, batch_size=32)

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

    __check_dir_exists("models")

    model.save_pretrained("./models/classifier")

    evaluate_classifier(dataset)


def evaluate_classifier(dataset):

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    def tokenize(examples):
        tokens = tokenizer(examples["premise_hypothesis"],
                           padding="max_length", max_length=250, truncation=True)
        tokens["labels"] = examples["label"]
        return tokens

    tokenized_dataset = dataset["test"].map(
        tokenize, batched=True, load_from_cache_file=False, num_proc=4, remove_columns=dataset["test"].column_names)
    tokenized_dataset.set_format(
        "torch", columns=['input_ids', 'attention_mask', 'labels'])

    eval_dataloader = DataLoader(
        tokenized_dataset, batch_size=32)

    model = RobertaForSequenceClassification.from_pretrained(
        "./models/classifier", num_labels=3)

    model.to(device)

    model.eval()

    progress_bar = tqdm(range(len(eval_dataloader)))

    nli_pred = []
    nli_gold = []

    for batch in eval_dataloader:

        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():

            outputs = model(**batch)

        logits = outputs.logits

        nli_pred.extend(torch.argmax(logits, dim=-1))

        nli_gold.extend(batch["labels"])

        progress_bar.update(1)

    print(classification_report(y_pred=nli_pred, y_true=nli_gold, target_names=["entailment", "neutral", "contradiction"]))


def train_explanator(dataset):

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    classifier_model = RobertaForSequenceClassification.from_pretrained(
        "./models/classifier", num_labels=3)
    classifier_model.to(device)

    def tokenize(examples):
        processed = {}
        tokens = tokenizer(examples["premise_hypothesis"], padding="max_length",
                           max_length=250, truncation=True, return_tensors="pt")

        processed["labels"] = tokenizer(examples["explanation_1"], padding="max_length",
                                        max_length=250, truncation=True)["input_ids"]
        
        tokens = {k: v.to(classifier_model.device) for k, v in tokens.items()}

        encoder_output =  classifier_model(
            **tokens, output_hidden_states=True)["hidden_states"]
        processed["encoder_hidden_states"] = encoder_output[-1].cpu().detach().numpy()
        processed["inputs_embeds"] = encoder_output[0].cpu().detach().numpy()
        return processed

    tokenized_dataset = dataset["train"].map(
        tokenize, batched=True, batch_size=32, load_from_cache_file=False, remove_columns=dataset["train"].column_names)
    tokenized_dataset.set_format(
        "torch", columns=['encoder_hidden_states', 'labels', "inputs_embeds"])
    
    def collator(batch):
        new_batch = {}
        for sample in batch:
            sample["encoder_hidden_states"] = torch.stack(sample["encoder_hidden_states"])
            sample["inputs_embeds"] = torch.stack(sample["inputs_embeds"])

        new_batch["labels"] = torch.stack([s["labels"] for s in batch])
        new_batch["encoder_hidden_states"] = torch.stack([s["encoder_hidden_states"] for s in batch])
        new_batch["inputs_embeds"] = torch.stack([s["inputs_embeds"] for s in batch])
        return new_batch

    train_dataloader = DataLoader(
        tokenized_dataset, shuffle=True, batch_size=32, collate_fn=collator)

    explanation_model = RobertaForCausalLM.from_pretrained(
        "roberta-base", is_decoder=True, add_cross_attention=True)
    explanation_model.to(device)

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

    __check_dir_exists("models")

    explanation_model.save_pretrained("./models/explanator")


def __check_dir_exists(dir: str):
    path = str(dir)
    if not os.path.exists(path):
        os.makedirs(path)
