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

### Dependencies

All dependecies needed to run our scripts can be installed by running the following command:

```
pip3 install -r requirements.txt
```

### Pre-trained models

Training our models can take up a significant amount of time, therefore we recommend to download our models
by using the following link:

TODO:
[Pre-trained Models](google colab link)

## GitHub Structure

We have divided our GitHub repository into three main directories, which are listed below.

````
.
├── data_analysis                           # Files used for data analysis              
├── model_training                    
│   ├── explain_from_classification         # Model training scripts for the Explain from Classification task      
│   ├── predict_and_explain                 # Model training scripts for the Predict and Explain task
│   └── predict_using_explanations          # Model training scripts for the Predict using Explanations task
├── peregrine_scripts                       # Peregrine job scripts                     
├── .gitignore                    
├── .pylintrc                   
├── LICENSE
└── README.md
└── requirements.txt
````
