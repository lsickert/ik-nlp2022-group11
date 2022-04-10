# Natural Language Processing : Final Project

**Project 5: Evaluating Human Explanations in Natural Language Inference**

TODO:
[Project Report](www.google.com)

## Group information

**Group number:** 11

**Authors:**

- Folkert Leistra    (f.a.leistra@student.rug.nl)
- Ludwig Sickert     (l.m.sickert@student.rug.nl)
- Thijmen Dam        (t.m.dam@student.rug.nl)

## How to use

### Sub-tasks

We  have divided our project into three sub-tasks, these are:

- Explain from Classification (also includes baseline NLI classification model)
- Predict and Explain
- Predict using Explanation

For each of these sub-tasks, we have created a directory which can be found in the GitHub structure overview below.

**The scripts that are present in these directories are used to do the following steps:**

1. Load all the data from HuggingFace
2. Load the tokenizer and model from HuggingFace
3. Preprocess the data
4. Train on the training data
5. Validate on the validation data
6. Evaluate on the test set

In order to train the models on the peregrine cluster, the provided jobscript `peregrine_script.sh` can be used by replacing the last line in the script with the respective model to be trained.

All scripts for the sub-tasks are automatically evaluated and output the relevant evaluation metrics.

### Dependencies

All dependecies needed to run our scripts can be installed by running the following command:

```
pip3 install -r requirements.txt
```

### Pre-trained models

Training our models can take up a significant amount of time, therefore we recommend to download our models
by using the following link:

[Pre-trained Models](https://drive.google.com/drive/folders/1BQPve4I38Zvn2Cb1xlBJT7vgzeHw8WNS?usp=sharing)

### Predicting on Unseen Data

In order to user our pre-trained models on your own custom dataset, you can use the training scripts
that can be found in the directories of our sub-tasks.

There are two main things to keep in mind when running on your own dataset:

#### 1: Change the data source

The models have been trained on the [E-SNLI](https://huggingface.co/datasets/esnli) dataset, loaded from hugginface. Therefore, when using your own data, you have to make sure that it contains the following columns:


premise (string) | hypothesis (string) | label (class label)     | explanation_1 (string)
---------------- |------------------|-------------------------| ----------
This church choir sings to the masses ....| The church has cracks in the ceiling.                 | 1 (neutral)             | Not all churches have cracks in the ....

Detailed information about the columns can be found [here](https://huggingface.co/datasets/esnli).

You can load in your own data by using any of the scripts in the sub-task directories and changing  the following lines:

```python
from transformers import datasets
# Loading the E-SNLI dataset from HuggingFace
train = datasets.load_dataset('esnli', split='train').shuffle(seed)
val = datasets.load_dataset('esnli', split='validation').shuffle(seed)
test = datasets.load_dataset('esnli', split='test').shuffle(seed)
```

Important: in order to use your own data, you need to keep the variables the same name.

#### 2: Load in your/our custom pre-trained model

Now that you have changed the path to the model data, you need to initialize the pre-trained model and tokenzier. This can be achieved by changing the following lines:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-base')
```

## GitHub Structure

We have divided our GitHub repository into three main directories, which are listed below.

````
.
├── data_analysis                           # scripts used for data analysis
|   ├── pickles                             # pickle files used for data analysis           
├── model_training                    
│   ├── explain_from_classification         # Model training scripts for the Explain from Classification task      
│   ├── predict_and_explain                 # Model training scripts for the Predict and Explain task
│   └── predict_using_explanations          # Model training scripts for the Predict using Explanations task
├── peregrine_scripts                       # Peregrine job script                     
├── .gitignore                    
├── .pylintrc                   
├── LICENSE
└── README.md
└── requirements.txt
````
