from datasets import load_dataset

def load_data():
    data = load_dataset("esnli")
    return data