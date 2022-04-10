from datasets import load_dataset

text_labels = ["entailment", "neutral", "contradiction"]


def add_columns(row):
    row["text_label"] = text_labels[row["label"]]
    row["premise_hypothesis"] = f'{row["premise"]}</s>{row["hypothesis"]}'
    row["prem_hypo_expl"] = f'{row["premise"]}</s>{row["hypothesis"]}</s>{row["explanation_1"]}'
    row["label_explanation_1"] = f'{row["text_label"]} {row["explanation_1"]}'

    if len(row["explanation_2"]) > 0:
        row["label_explanation_2"] = f'{row["text_label"]} {row["explanation_2"]}'

    if len(row["explanation_3"]) > 0:
        row["label_explanation_3"] = f'{row["text_label"]} {row["explanation_3"]}'

    return row


def load_data(dataset: str = None):
    '''
    returns the train-test-validation split of the dataset.
    if the optional parameter "dataset" is set, it returns only that part of the dataset
    '''
    data = load_dataset("esnli")

    proc_data = data.map(add_columns, num_proc=4)
    if dataset == None:
        return proc_data
    else:
        return proc_data[dataset]


def load_data_as_df(dataset: str):
    '''
    returns a cleaned Pandas dataframe of one of the three dataset splits.
    '''
    data = load_data(dataset)

    df = data.to_pandas()

    df["premise"] = df["premise"].astype("string")
    df["hypothesis"] = df["hypothesis"].astype("string")
    df["text_label"] = df["text_label"].astype("string")
    df["standalone_hypothesis"] = df["standalone_hypothesis"].astype("string")
    df["premise_hypothesis"] = df["premise_hypothesis"].astype("string")
    df["explanation_1"] = df["explanation_1"].astype("string")
    df["label_explanation_1"] = df["label_explanation_1"].astype("string")

    if dataset == "train":
        df.drop(columns=["explanation_2", "explanation_3"], inplace=True)
    else:
        df["explanation_2"] = df["explanation_2"].astype("string")
        df["explanation_3"] = df["explanation_3"].astype("string")
        df["label_explanation_2"] = df["label_explanation_2"].astype("string")
        df["label_explanation_3"] = df["label_explanation_3"].astype("string")

    return df
