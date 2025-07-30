# train_model.py
import pandas as pd
import re
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Text cleaner
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower()

# Load and prepare data
df = pd.read_csv("C:/Users/utkar/Downloads/archive (1)/UpdatedResumeDataSet.csv")
df["Resume"] = df["Resume"].apply(clean_text)

le = LabelEncoder()
df["label"] = le.fit_transform(df["Category"])
categories = list(le.classes_)
pd.Series(categories).to_csv("label_classes.csv", index=False)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df["Resume"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# Tokenize
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=512)
train_labels = y_train.tolist()

class ResumeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }

train_dataset = ResumeDataset(train_encodings, train_labels)

# Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(categories))

# Training config
args = TrainingArguments(
    output_dir="./bert_resume_model",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    logging_dir='./logs',
    save_strategy="no"
)

trainer = Trainer(model=model, args=args, train_dataset=train_dataset)
trainer.train()

# Save model and tokenizer
model.save_pretrained("./bert_resume_model")
tokenizer.save_pretrained("./bert_resume_model")
