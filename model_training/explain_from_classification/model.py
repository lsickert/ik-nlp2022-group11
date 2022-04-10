import torch
import os

from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, RobertaForCausalLM, RobertaModel, AutoTokenizer, AutoModelForSeq2SeqLM, get_scheduler

from torch.utils.data import DataLoader
from torch.optim import AdamW

from tqdm.auto import tqdm

from sklearn.metrics import classification_report

from data_analysis import data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def train_classifier(dataset):
    """
    Train the baseline NLI classification model
    """

    tokenized_dataset = __classifier_tokenize(dataset["train"])

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
    """
    Evaluate the baseline NLI classification model
    """

    tokenized_dataset = __classifier_tokenize(dataset["test"])

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

        nli_pred.extend(torch.argmax(logits, dim=-1).tolist())

        nli_gold.extend(batch["labels"])

        progress_bar.update(1)

    print(classification_report(y_pred=nli_pred, y_true=nli_gold,
          target_names=["entailment", "neutral", "contradiction"]))


def train_explanator(dataset):
    """
    Train the explanation generator using the hidden states from the classification model as input
    """

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    classifier_model = RobertaModel.from_pretrained(
        "./models/classifier")
    classifier_model.to(device)

    def tokenize(examples):
        processed = {}
        tokens = tokenizer(examples["premise_hypothesis"], padding="max_length",
                           max_length=193, truncation=True, return_tensors="pt")

        processed["labels"] = tokenizer(examples["explanation_1"], padding="max_length",
                                        max_length=193, truncation=True)["input_ids"]

        tokens = {k: v.to(classifier_model.device) for k, v in tokens.items()}

        encoder_output = classifier_model(
            **tokens, output_hidden_states=True)
        processed["encoder_hidden_states"] = encoder_output.last_hidden_state.cpu(
        ).detach().numpy()
        processed["input_ids"] = tokens["input_ids"].cpu(
        ).detach().numpy()
        processed["encoder_attention_mask"] = tokens["attention_mask"].cpu(
        ).detach().numpy()
        return processed

    tokenized_dataset = dataset["train"].map(
        tokenize, batched=True, batch_size=32, remove_columns=dataset["train"].column_names)
    tokenized_dataset.set_format(
        "torch", columns=['encoder_hidden_states', 'encoder_attention_mask', 'labels', "input_ids"])

    train_dataloader = DataLoader(
        tokenized_dataset, shuffle=True, batch_size=32, collate_fn=__explanation_collator)

    explanation_model = RobertaForCausalLM.from_pretrained(
        "roberta-base", is_decoder=True, add_cross_attention=True)
    explanation_model.to(device)

    optimizer = AdamW(explanation_model.parameters(), lr=1e-5)

    num_epochs = 3

    num_training_steps = num_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))

    explanation_model.train()

    for epoch in range(num_epochs):
        batch_c = 0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = explanation_model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.set_postfix({"loss": loss.item()})
            progress_bar.update(1)

            if(batch_c == 50):
                tqdm.write(f"loss: {loss.item()}")
                batch_c = 0
            else:
                batch_c += 1

    __check_dir_exists("models")

    explanation_model.save_pretrained("./models/explanator")


def predict_single(sentence):
    """
    Generate a single classification and explanation
    """

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    class_tokens = tokenizer(sentence, padding="max_length",
                             max_length=125, truncation=True, return_tensors="pt")

    classifier_model = RobertaForSequenceClassification.from_pretrained(
        "./models/classifier", num_labels=3)
    classifier_model.to(device)

    class_tokens.to(device)

    with torch.no_grad():

        class_outputs = classifier_model(
            **class_tokens, output_hidden_states=True)

    logits = class_outputs.logits

    pred = torch.argmax(logits, dim=-1).item()

    print(f"classification: {pred}")

    expl_tokens = {}
    expl_tokens["encoder_hidden_states"] = class_outputs["hidden_states"][-1].cpu().detach()
    expl_tokens["inputs_embeds"] = class_outputs["hidden_states"][0].cpu().detach()
    #expl_tokens["input_ids"] = class_tokens["input_ids"]
    #expl_tokens["encoder_attention_mask"] = class_tokens["attention_mask"]

    explanation_model = RobertaForCausalLM.from_pretrained(
        "./models/explanator")
    explanation_model.to(device)

    outputs = explanation_model(**expl_tokens)

    out_tokens = torch.argmax(outputs.logits, dim=2)

    print(tokenizer.batch_decode(out_tokens, skip_special_tokens=True))


def __explanation_collator(batch):
    new_batch = {}
    for sample in batch:
        sample["encoder_hidden_states"] = torch.stack(
            sample["encoder_hidden_states"])

    new_batch["labels"] = torch.stack([s["labels"] for s in batch])
    new_batch["encoder_hidden_states"] = torch.stack(
        [s["encoder_hidden_states"] for s in batch])
    new_batch["input_ids"] = torch.stack(
        [s["input_ids"] for s in batch])
    new_batch["encoder_attention_mask"] = torch.stack(
        [s["encoder_attention_mask"] for s in batch])
    return new_batch


def __classifier_tokenize(dataset):

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    def tokenize(examples):
        tokens = tokenizer(examples["premise_hypothesis"],
                           padding="max_length", max_length=250, truncation=True)
        tokens["labels"] = examples["label"]
        return tokens

    tokenized_dataset = dataset.map(
        tokenize, batched=True, num_proc=4, remove_columns=dataset.column_names)
    tokenized_dataset.set_format(
        "torch", columns=['input_ids', 'attention_mask', 'labels'])

    return tokenized_dataset


def __check_dir_exists(dir: str):
    path = str(dir)
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    ds = data.load_data()

    train_classifier(ds)

    train_explanator(ds)
