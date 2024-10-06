import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    get_scheduler,
)
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler


class TextProcess():
    def __init__(self, label_columns, test_infer=False, length_infer=300, data_name=None, infer_on_unlabeled=False):
        
        # please select the label_columns based on the table I listed below
        if infer_on_unlabeled:
            self.df = pd.read_excel("./dataset/"+data_name)
        else:
            self.df = pd.read_excel("./dataset/1003_data_cleaned.xlsx")
        self.comments = self.df["text"].astype(str)
        if isinstance(label_columns, int):
            if test_infer:
                self.selected_columns = self.df.iloc[0:length_infer, label_columns:label_columns+1]              # 2
            else:
                self.selected_columns = self.df.iloc[:, label_columns:label_columns+1]
        if isinstance(label_columns, list):
            if test_infer:
                self.selected_columns = self.df.iloc[0:length_infer, label_columns[0]:label_columns[1]]          # 2:12
            else:
                self.selected_columns = self.df.iloc[:, label_columns[0]:label_columns[1]]              # 2:12
        self.labels_array = self.selected_columns.to_numpy().astype(float)
        if test_infer:
            self.comments_list = self.comments.tolist()[0:length_infer]
        else:
            self.comments_list = self.comments.tolist()

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]).to('cuda') for key, val in self.encodings.items()}

    def __len__(self):
        return self.encodings['labels'].shape[0]

# We set the model into 1 output
"""
data selection from raw data file, you should choose index from [2: 12] and modify the input data by modifying the label_columns.
AI_Usage:                           2
Humanlikeness_Mental                3
Humanlikeness_Visual                4
PSI_Object_AI                       5
PSI_object_charactor_itself         6
PSI_Agreement                       7
PSI_Eexpress_opinion                8
PSI_group                           9
PSI_intersted                       10
AI_Merged                           11
"""
def prepare_data(label_columns, device):
    text_process = TextProcess(label_columns=label_columns)
    X_train, X_test, y_train, y_test = train_test_split(text_process.comments_list, text_process.labels_array, test_size=0.2, random_state=42)

    print(f"We have done data loading! and the length of them are: {len(X_train)}, {len(y_test)}. We start to load tokenizer")

    model_name = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model = model.to(device)
    print(f"We have loaded tokenizer and will use it to convert data")

    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128, return_tensors='pt')
    train_encodings['labels'] = torch.from_numpy(y_train)

    test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128, return_tensors='pt')
    test_encodings['labels'] = torch.from_numpy(y_test)

    train_dataset, test_dataset = TextDataset(train_encodings), TextDataset(test_encodings)
    print("We have converted them and return it to trainer")

    return train_dataset, test_dataset, model

def prepare_data_without_model(label_columns, device, test_infer=True, length_infer=300, debug=False, data_name=None, infer_on_unlabeled=False):
    if infer_on_unlabeled:
        test_infer=False
    text_process = TextProcess(label_columns=label_columns, test_infer=test_infer, length_infer=300, data_name=data_name, infer_on_unlabeled=infer_on_unlabeled)
    x_test, y_test = text_process.comments_list, text_process.labels_array

    print(f"We have done data loading! and the length of them are: {len(x_test), len(y_test)}. We start to load tokenizer")

    model_name = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=128, return_tensors='pt')
    test_encodings['labels'] = torch.from_numpy(y_test)

    test_dataset = TextDataset(test_encodings)
    print("We have converted them and return it to trainer")

    if debug:
        return test_dataset, x_test
    else:
        return test_dataset


if __name__=="__main__":
    _1, y_test_1 = prepare_data_without_model(label_columns=2,device="cuda:5",debug=True)
    _2, y_test_3 = prepare_data_without_model(label_columns=3,device="cuda:5",debug=True)
    _3, y_test_2 = prepare_data_without_model(label_columns=4,device="cuda:5",debug=True)

    print(y_test_1==y_test_2, y_test_1==y_test_3, y_test_3==y_test_2)

    test_loader_1 = DataLoader(_1, batch_size=100, sampler=SequentialSampler(_1))
    test_loader_2 = DataLoader(_2, batch_size=100, sampler=SequentialSampler(_2))
    test_loader_3 = DataLoader(_3, batch_size=100, sampler=SequentialSampler(_3))

    for (b1, b2, b3) in zip(test_loader_1, test_loader_2, test_loader_3):
        print(b1['labels']==b2['labels'])
        print(b1['labels']==b3['labels'])
        print(b3['labels']==b2['labels'])