import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split 
import pandas as pd
import numpy as np
import random
import nlpaug.augmenter.word as naw
import gc

def clean_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def make_model_contiguous(model):
    for name, param in model.named_parameters():
        if not param.data.is_contiguous():
            param.data = param.data.contiguous()
            print(f"Made {name} contiguous.")

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, augment=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment
        self.aug_word = naw.SynonymAug(aug_src='wordnet')

    def __len__(self):
        return len(self.texts)

    def augment_text(self, text):
        if self.augment and random.random() > 0.7:
            return self.aug_word.augment(text)
        return text

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        if self.augment:
            text = self.augment_text(text)
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

df = pd.read_excel('/home/WenqiQiu/bilibili/bert_training/bert_training/0904_data.xlsx')
df = df.dropna(subset=['text'])
texts = df['text'].values
labels = df.drop(columns=['text']).values.astype(np.float32)

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)

local_model_directory = '/home/WenqiQiu/.cache/modelscope/hub/iic/nlp_roberta_backbone_large_std'
tokenizer = BertTokenizer.from_pretrained(local_model_directory)
model = BertForSequenceClassification.from_pretrained(local_model_directory, num_labels=labels.shape[1])

train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len=128, augment=True)
val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_len=128)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=RandomSampler(train_dataset))
val_loader = DataLoader(val_dataset, batch_size=32, sampler=SequentialSampler(val_dataset))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=3e-5)
scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 100)

def train_epoch(model, data_loader, optimizer, criterion, device, scheduler):
    model.train()
    total_loss = 0
    for batch in data_loader:
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(inputs, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)

def validate_epoch(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(inputs, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            preds = torch.sigmoid(outputs.logits).cpu().numpy() 
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"Predictions shape: {all_preds.shape}, Labels shape: {all_labels.shape}")

    rounded_preds = np.round(all_preds)

    try:
        f1 = f1_score(rounded_preds, all_labels, average='samples')  
    except ValueError as e:
        print(f"Error in F1 Score calculation: {e}")
        f1 = 0

    try:
        auc = roc_auc_score(all_labels, all_preds, average='macro') 
    except ValueError as e:
        print(f"Error in AUC Score calculation: {e}")
        auc = 0

    try:
        accuracy = accuracy_score(rounded_preds, all_labels)
    except ValueError as e:
        print(f"Error in Accuracy Score calculation: {e}")
        accuracy = 0

    report = classification_report(all_labels, rounded_preds, target_names=[f'Label {i}' for i in range(all_labels.shape[1])], output_dict=True, zero_division=0)
    print(report)

    avg_loss = total_loss / len(data_loader)

    return avg_loss, f1, accuracy, auc



best_auc = 0
for epoch in range(100):
    print(f"Epoch {epoch + 1}/100")
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
    val_loss, val_f1, val_accuracy, val_auc = validate_epoch(model, val_loader, criterion, device)
    print(f"Training Loss: {train_loss}, Validation Loss: {val_loss}, F1 Score: {val_f1}, Accuracy: {val_accuracy}, AUC: {val_auc}")
    
    if val_auc > best_auc:
        best_auc = val_auc
        model_to_save = model.module if hasattr(model, 'module') else model
        make_model_contiguous(model_to_save) 
        model_to_save.save_pretrained('/home/WenqiQiu/bilibili/bert_training/bert_training/best_model')
        tokenizer.save_pretrained('/home/WenqiQiu/bilibili/bert_training/bert_training/best_model')

make_model_contiguous(model)
model.save_pretrained('/home/WenqiQiu/bilibili/bert_training/bert_training/saved_model')
tokenizer.save_pretrained('/home/WenqiQiu/bilibili/bert_training/bert_training/saved_model')
