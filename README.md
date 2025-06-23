# blackhumouranalysis
# TASK 1 - VISIONE (SENTIMENT DA IMMAGINE)

# Import librerie

!pip install -q torch torchvision pandas scikit-learn matplotlib pillow tqdm

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.optim as optim
#Carica il dataset
from google.colab import drive
drive.mount('/content/drive')

CSV_PATH = "/content/drive/MyDrive/DA4B/memotion_dataset_7k/labels.csv"
IMG_DIR = "/content/drive/MyDrive/DA4B/memotion_dataset_7k/images"

df = pd.read_csv(CSV_PATH)
df = df[['image_name', 'overall_sentiment']]
df.dropna(inplace=True)

# Mappa da stringa a numero, se necessario
if df['overall_sentiment'].dtype == 'object':
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['overall_sentiment'] = df['overall_sentiment'].map(label_map)

# Tieni solo le etichette valide
df = df[df['overall_sentiment'].isin([0, 1, 2])].copy()

# Verifica
print("Etichette presenti dopo pulizia:", df['overall_sentiment'].value_counts())
df.head()
df['overall_sentiment'].value_counts().plot(kind='bar', title='Distribuzione classi')
plt.show()

output_dir = "/content/drive/MyDrive/DA4B/memotion_dataset_7k/memes_balanced"
os.makedirs(output_dir, exist_ok=True)

sample_size = 1000
balanced_df = pd.DataFrame()

for label in [0, 1, 2]:
    class_df = df[df['overall_sentiment'] == label]
    class_df = class_df.sample(n=min(sample_size, len(class_df)), random_state=42)


    if len(class_df) < sample_size:
        extra = sample_size - len(class_df)
        dup_df = class_df.sample(n=extra, replace=True, random_state=42)

        flipped_rows = []
        for _, row in dup_df.iterrows():
            img_path = os.path.join(IMG_DIR, row['image_name'])
            new_name = f"flip_{row['image_name']}"

            try:
                img = Image.open(img_path).convert("RGB")
                flipped = ImageOps.mirror(img)
                flipped.save(os.path.join(output_dir, new_name))

                flipped_rows.append({
                    'image_name': new_name,
                    'overall_sentiment': row['overall_sentiment']
                })

            except Exception as e:
                print(f"Errore con immagine: {img_path} → {e}")
                continue

        dup_df = pd.DataFrame(flipped_rows)
        class_df = pd.concat([class_df, dup_df], ignore_index=True)

    for _, row in class_df.iterrows():
        src = os.path.join(IMG_DIR, row['image_name'])
        dst = os.path.join(output_dir, row['image_name'])

        if not os.path.exists(dst):
            try:
                img = Image.open(src).convert("RGB")
                img.save(dst)
            except Exception as e:
                print(f"Errore salvataggio immagine: {src} → {e}")
                continue

    balanced_df = pd.concat([balanced_df, class_df], ignore_index=True)

print("Distribuzione dopo il bilanciamento:")
print(balanced_df['overall_sentiment'].value_counts())
#Dataset e DataLoader
train_val_df, test_df = train_test_split(
    balanced_df,
    test_size=0.15,
    stratify=balanced_df['overall_sentiment'],
    random_state=42
)

# Dataset e DataLoader
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.1765,  # ~15% of original
    stratify=train_val_df['overall_sentiment'],
    random_state=42
)

class MemeDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['image_name']
        label = self.dataframe.iloc[idx]['overall_sentiment']
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path)

        if image.mode in ['P', 'RGBA']:
            image = image.convert("RGBA")
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image).convert("RGB")
        else:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, int(label)


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

IMG_DIR = "/content/drive/MyDrive/DA4B/memotion_dataset_7k/memes_balanced"

train_dataset = MemeDataset(train_df, IMG_DIR, transform=train_transform)
val_dataset = MemeDataset(val_df, IMG_DIR, transform=val_test_transform)
test_dataset = MemeDataset(test_df, IMG_DIR, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Modello (EfficientNet-B0 + Transfer Learning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torchvision.models as models
import torch.nn as nn

# Carica EfficientNet-B0 preaddestrata
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)

# Sostituisci il classificatore finale con uno per 4 classi
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Addestramento
EPOCHS = 10
PATIENCE = 4
BEST_MODEL_PATH = 'best_model.pth'

best_val_acc = 0
epochs_no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # 🔍 Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    print(f"Epoch {epoch+1}/{EPOCHS} — Train Loss: {avg_train_loss:.4f} — Val Loss: {avg_val_loss:.4f} — Val Accuracy: {val_accuracy:.2f}%")

    # 💾 Salva il miglior modello
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"✅ Saved new best model with accuracy: {best_val_acc:.2f}%")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"⚠️ No improvement. Patience: {epochs_no_improve}/{PATIENCE}")
        if epochs_no_improve >= PATIENCE:
            print("⏹️ Early stopping triggered.")
            break

# Valutazione finale sul test set
model.load_state_dict(torch.load('best_model.pth'))
model = model.to(device)
model.eval()

correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = 100 * correct / total
print(f"✅ Test Accuracy: {test_accuracy:.2f}%")

# Matrice di confusione
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["very_negative", "negative", "positive", "very_positive"])
disp.plot(cmap='Blues')
plt.title("Test Set Confusion Matrix")
plt.show()

# TASK 2 - MULTISTASK NLP (TESTO + MULTILABEL + SENTIMENT)

# Installazione librerie
!pip install transformers imbalanced-learn scikit-learn seaborn matplotlib --quiet

# Import
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
import time

# FocalLoss per classificazione multiclass
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, input, target):
        ce_loss = self.ce(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Dataset
class MemeDataset(Dataset):
    def __init__(self, texts, multilabels, sentiments, tokenizer, max_len=128):
        self.texts = list(texts)
        self.multilabels = multilabels
        self.sentiments = sentiments
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'multilabels': torch.tensor(self.multilabels[idx], dtype=torch.float),
            'sentiment': torch.tensor(self.sentiments[idx], dtype=torch.long)
        }

# Modello multitask
class RobertaMultitaskClassifier(nn.Module):
    def __init__(self, num_multilabels=4, num_sentiment_classes=5):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        hidden_size = self.roberta.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.multilabel_head = nn.Linear(hidden_size, num_multilabels)
        self.sentiment_head = nn.Linear(hidden_size, num_sentiment_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        return self.multilabel_head(pooled), self.sentiment_head(pooled)

# Caricamento dati
df = pd.read_csv("/content/drive/MyDrive/Università/DA4B/memotion_dataset_7k/labels.csv")
df['text_corrected'] = df['text_corrected'].fillna("").astype(str)

# Encoding
multilabel_cols = ['humour', 'sarcasm', 'offensive', 'motivational']
for col in multilabel_cols:
    df[col] = LabelEncoder().fit_transform(df[col])
le_sentiment = LabelEncoder()
df['overall_sentiment'] = le_sentiment.fit_transform(df['overall_sentiment'])

X = df['text_corrected']
y_multi = df[multilabel_cols].values
y_sent = df['overall_sentiment'].values

# Train / Val / Test Split con Oversampling
X_train, X_temp, y_multi_train, y_multi_temp, y_sent_train, y_sent_temp = train_test_split(
    X, y_multi, y_sent, test_size=0.3, stratify=y_sent, random_state=42)
X_val, X_test, y_multi_val, y_multi_test, y_sent_val, y_sent_test = train_test_split(
    X_temp, y_multi_temp, y_sent_temp, test_size=0.5, stratify=y_sent_temp, random_state=42)

ros = RandomOverSampler(random_state=42)
X_train_os, y_sent_train_os = ros.fit_resample(X_train.values.reshape(-1, 1), y_sent_train)
y_multi_train_os = y_multi_train[ros.sample_indices_]
X_train_os = X_train_os.flatten()

# Tokenizer e DataLoader
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
train_dataset = MemeDataset(X_train_os, y_multi_train_os, y_sent_train_os, tokenizer)
val_dataset = MemeDataset(X_val, y_multi_val, y_sent_val, tokenizer)
test_dataset = MemeDataset(X_test, y_multi_test, y_sent_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

# Setup training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RobertaMultitaskClassifier().to(device)

loss_fn_multi = nn.BCEWithLogitsLoss()
loss_fn_sent = FocalLoss(gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Funzioni training e valutazione
def train_epoch(model, loader):
    model.train()
    total_loss = 0
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        multilabels = batch['multilabels'].to(device)
        sentiment = batch['sentiment'].to(device)

        optimizer.zero_grad()
        logits_multi, logits_sent = model(input_ids, attention_mask)
        loss = loss_fn_multi(logits_multi, multilabels) + loss_fn_sent(logits_sent, sentiment)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    preds_s, trues_s = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment = batch['sentiment'].to(device)
            _, logits_sent = model(input_ids, attention_mask)
            pred_sent = torch.argmax(logits_sent, dim=-1).cpu().numpy()
            preds_s.extend(pred_sent)
            trues_s.extend(sentiment.cpu().numpy())
    return preds_s, trues_s

# Addestramento
best_f1 = 0
patience, counter = 3, 0
EPOCHS = 10
for epoch in range(EPOCHS):
    start = time.time()
    train_loss = train_epoch(model, train_loader)
    preds, trues = evaluate(model, val_loader)
    f1 = f1_score(trues, preds, average='macro')
    print(f"\n🔁 Epoch {epoch+1} - Loss: {train_loss:.4f} - Macro F1: {f1:.4f} - Time: {time.time() - start:.1f}s")

    if f1 > best_f1:
        best_f1 = f1
        counter = 0
        torch.save(model.state_dict(), "best_roberta_model.pt")
    else:
        counter += 1
        if counter >= patience:
            print("⏹️ Early stopping triggered.")
            break

# Test finale
model.load_state_dict(torch.load("best_roberta_model.pt"))
model.eval()
preds, trues = evaluate(model, test_loader)
print("\n📌 FINAL TEST REPORT")
print(classification_report(trues, preds, target_names=le_sentiment.classes_, zero_division=0))

# Confusion matrix
cm = confusion_matrix(trues, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le_sentiment.classes_, yticklabels=le_sentiment.classes_)
plt.title("Confusion Matrix - Sentiment")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix_roberta.png")
plt.show()

# TASK 3 - CLASSIFICAZIONE DELLA NUOVA FEATURE BLACK HUMOUR E CONFRONTO CON LA FEATURE SARCASMO
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import os

# Leggi il file CSV
CSV_PATH = '/content/drive/MyDrive/DA4B dataset/memotion_dataset_7k/labels.csv'
df = pd.read_csv(CSV_PATH)
IMAGES_DIR = '/content/drive/MyDrive/DA4B dataset/memotion_dataset_7k/images'
IMAGES_FILES = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png'))]

# Mostra le prime righe del DataFrame
df.head()

# Mostra i primi 5 file trovati
IMAGES_FILES[:5]

import pandas as pd

df = pd.read_csv(CSV_PATH)
print(df['sarcasm'].value_counts())
print(df['sarcasm'].value_counts(normalize=True))  # per le percentuali

import pandas as pd
from sklearn.model_selection import train_test_split

# Definisci la mappatura delle etichette sarcasmo
label_mapping = {
    'general': 2,
    'very_twisted': 1,
    'twisted_meaning': 1,
    'not_sarcastic': 0
}

# Applica la mappatura sull'intero dataframe
df['sarcasm_level'] = df['sarcasm'].map(label_mapping)

# Suddividi in train (80%) e temp (20%)
train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['sarcasm_level']
)

# Suddividi temp in val (10%) e test (10%)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df['sarcasm_level']
)

# Controlla le distribuzioni
print("Train sarcasm_level counts:\n", train_df['sarcasm_level'].value_counts())
print("Validation sarcasm_level counts:\n", val_df['sarcasm_level'].value_counts())
print("Test sarcasm_level counts:\n", test_df['sarcasm_level'].value_counts())

from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Prepara i testi (rimuovi i NaN e converti a stringhe)
train_texts_sarcasm = train_df['text_corrected'].fillna("").astype(str).tolist()
val_texts_sarcasm = val_df['text_corrected'].fillna("").astype(str).tolist()
test_texts_sarcasm = test_df['text_corrected'].fillna("").astype(str).tolist()

# Prepara le etichette numeriche
train_labels_int_sarcasm = train_df['sarcasm_level'].astype(int).tolist()
val_labels_int_sarcasm = val_df['sarcasm_level'].astype(int).tolist()
test_labels_int_sarcasm = test_df['sarcasm_level'].astype(int).tolist()

# One-hot encoding (se richiesto dal modello)
train_labels_sarcasm = to_categorical(train_labels_int_sarcasm, num_classes=3)
val_labels_sarcasm = to_categorical(val_labels_int_sarcasm, num_classes=3)
test_labels_sarcasm = to_categorical(test_labels_int_sarcasm, num_classes=3)

# Inizializza il tokenizer BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenizza i testi
train_encodings_sarcasm = tokenizer(train_texts_sarcasm, padding=True, truncation=True, max_length=128, return_tensors='tf')
val_encodings_sarcasm = tokenizer(val_texts_sarcasm, padding=True, truncation=True, max_length=128, return_tensors='tf')
test_encodings_sarcasm = tokenizer(test_texts_sarcasm, padding=True, truncation=True, max_length=128, return_tensors='tf')

print("✅ Dati sarcasmo pronti: encoding + label mappate correttamente.")

# Dopo la tokenizzazione e l'elaborazione dei dati
train_inputs = {
    'input_ids': train_encodings_sarcasm['input_ids'],
    'attention_mask': train_encodings_sarcasm['attention_mask']
}
val_inputs = {
    'input_ids': val_encodings_sarcasm['input_ids'],
    'attention_mask': val_encodings_sarcasm['attention_mask']
}
print(train_inputs['input_ids'].shape)
print(train_inputs['attention_mask'].shape)

num_train_samples = len(train_df)
batch_size = 32
steps_per_epoch = np.ceil(num_train_samples / batch_size)
print(f"Batch per epoca: {steps_per_epoch}")

from transformers import TFBertForSequenceClassification
import tensorflow as tf
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint  # usa tf_keras per entrambi
import pickle
import os

# Crea cartella su Drive o locale
save_dir = '/content/drive/MyDrive/sarcasm_project'
os.makedirs(save_dir, exist_ok=True)

# Carica il modello BERT
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Compila
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# ModelCheckpoint
checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True,
    mode='min'
)

# History
history_sarc = model.fit(
    {'input_ids': train_encodings_sarcasm['input_ids'], 'attention_mask': train_encodings_sarcasm['attention_mask']},
    train_labels_sarcasm,
    validation_data=(
        {'input_ids': val_encodings_sarcasm['input_ids'], 'attention_mask': val_encodings_sarcasm['attention_mask']},
        val_labels_sarcasm
    ),
    epochs=5,
    batch_size=32,
    callbacks=[checkpoint] # Removed early_stopping from callbacks
)

# Valutazione
test_loss, test_accuracy = model.evaluate(
    {'input_ids': test_encodings_sarcasm['input_ids'], 'attention_mask': test_encodings_sarcasm['attention_mask']},
    test_labels_sarcasm
)

print(f"Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}")
model.save_pretrained('/content/drive/MyDrive/sarcasm_project/bert_model_phase1')

from transformers import TFBertForSequenceClassification
model_sarc = TFBertForSequenceClassification.from_pretrained('/content/drive/MyDrive/sarcasm_project/bert_model_phase1')

from transformers import TFBertForSequenceClassification

# Carica direttamente tutto il modello salvato (config + pesi)
model_sarc = TFBertForSequenceClassification.from_pretrained('/content/drive/MyDrive/sarcasm_project/bert_model_phase1')

# Mostra la struttura del modello
model_sarc.summary()

import pickle

# Percorso completo del file su Drive
path = '/content/drive/MyDrive/sarcasm_project/train_history.pkl'

with open(path, 'rb') as f:
    history = pickle.load(f)
print(history.keys())  # per vedere le chiavi disponibili

import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))

# Loss
plt.subplot(1,2,1)
plt.plot(history['loss'], label='Training Loss')
if 'val_loss' in history:
    plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy
plt.subplot(1,2,2)
if 'accuracy' in history:
    plt.plot(history['accuracy'], label='Training Accuracy')
if 'val_accuracy' in history:
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

import pandas as pd
CSV_NEW_PATH = '/content/drive/MyDrive/DA4B dataset/memotion_dataset_7k/labels.neww.csv'
df_raw = pd.read_csv(CSV_NEW_PATH, header=None)
df_split = df_raw[0].str.split(',', expand=True)
df_split.columns = [
    'id', 'image_name', 'text_ocr', 'text_corrected', 'humour',
    'sarcasm', 'offensive', 'motivational', 'overall_sentiment', 'level_black_humour', '""','""','""'
]
# Elimina la prima riga
df_split = df_split.iloc[1:]

# Elimina le ultime 3 colonne
df_split = df_split.iloc[:, :-3]

# Reimposta l'indice (opzionale)
df_split.reset_index(drop=True, inplace=True)

# Visualizza
print(df_split.head())
print(df_split.columns)
print (df_split.shape)

df_split.columns = df_split.columns.str.strip()  # Rimuove spazi prima/dopo
print(df_split['level_black_humour'].head(10))
CSV_NEW_PATH_OK = df_split
CSV_NEW_PATH_OK.head(10)
print(CSV_NEW_PATH_OK.shape)

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from transformers import BertTokenizer

df = df_split.copy()
df['level_black_humour'] = pd.to_numeric(df['level_black_humour'], errors='coerce')
df.dropna(subset=['level_black_humour'], inplace=True)
df['level_black_humour'] = df['level_black_humour'].astype(int)
df = df[df["level_black_humour"].isin([0, 1, 2])].reset_index(drop=True)
df['text_corrected'] = df['text_corrected'].fillna("")

tokenizer_black = BertTokenizer.from_pretrained('bert-base-uncased')
encodings_black = tokenizer(list(df['text_corrected']), truncation=True, padding=True, max_length=128)

labels = to_categorical(df['level_black_humour'], num_classes=3)

import numpy as np

input_ids = np.array(encodings_black['input_ids'])
attention_mask = np.array(encodings_black['attention_mask'])
labels = np.array(labels)

train_input_ids, val_input_ids, train_masks, val_masks, train_labels, val_labels = train_test_split(
    input_ids, attention_mask, labels,
    test_size=0.2, random_state=42, stratify=df['level_black_humour']
)

train_encodings_black = {
    'input_ids': tf.constant(train_input_ids),
    'attention_mask': tf.constant(train_masks)
}
val_encodings_black = {
    'input_ids': tf.constant(val_input_ids),
    'attention_mask': tf.constant(val_masks)
}

print(train_encodings_black['input_ids'].shape)
print(train_encodings_black['attention_mask'].shape)
print(val_encodings_black['input_ids'].shape)
print(val_encodings_black['attention_mask'].shape)
print("Data preparation for black humour model completed!")

print("Iniziali:", len(df_split))  # 6992
# Assicurati che la colonna sia numerica
df_split['level_black_humour'] = pd.to_numeric(df_split['level_black_humour'], errors='coerce')
print("Dopo to_numeric:", len(df_split))  # es. 6975

# Elimina righe con NaN in 'level_black_humour'
df_split = df_split.dropna(subset=['level_black_humour'])
print("Dopo dropna:", len(df_split))  # es. 6900

# Converti in interi
df_split['level_black_humour'] = df_split['level_black_humour'].astype(int)

# Tieni solo i valori 0,1,2
df_split = df_split[df_split['level_black_humour'].isin([0,1,2])]
print("Dopo filtro valori validi:", len(df_split))

# Reset index
df_split = df_split.reset_index(drop=True)

from sklearn.model_selection import train_test_split

# Suddividi in train+val e test
df_temp, test_df = train_test_split(
    df_split, test_size=0.15, random_state=42, stratify=df_split['level_black_humour'])

# Suddividi df_temp in train e val
train_df, val_df = train_test_split(
    df_temp, test_size=0.1765, random_state=42, stratify=df_temp['level_black_humour'])

print("Train size:", len(train_df))
print("Validation size:", len(val_df))
print("Test size:", len(test_df))

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from transformers import BertTokenizer
import tensorflow as tf

# Prepara i testi e le etichette per ogni split

train_texts_black = train_df['text_corrected'].fillna("").astype(str).tolist()
val_texts_black = val_df['text_corrected'].fillna("").astype(str).tolist()
test_texts_black = test_df['text_corrected'].fillna("").astype(str).tolist()

train_labels_black = to_categorical(train_df['level_black_humour'].astype(int), num_classes=3)
val_labels_black = to_categorical(val_df['level_black_humour'].astype(int), num_classes=3)
test_labels_black = to_categorical(test_df['level_black_humour'].astype(int), num_classes=3)

# Tokenizza testi con return_tensors='tf' per usare con TensorFlow
tokenizer_black = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings_black = tokenizer_black(
    train_texts_black, padding=True, truncation=True, max_length=128, return_tensors='tf')

val_encodings_black = tokenizer_black(
    val_texts_black, padding=True, truncation=True, max_length=128, return_tensors='tf')

test_encodings_black = tokenizer_black(
    test_texts_black, padding=True, truncation=True, max_length=128, return_tensors='tf')
print("Tokenizzazione black humour completata.")

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from transformers import BertTokenizer, TFBertForSequenceClassification
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy

# Prepara il DataFrame

df = df_split.copy()
df['level_black_humour'] = pd.to_numeric(df['level_black_humour'], errors='coerce')
df = df.dropna(subset=['level_black_humour'])
df['level_black_humour'] = df['level_black_humour'].astype(int)
df = df[df['level_black_humour'].isin([0,1,2])].reset_index(drop=True)
df['text_corrected'] = df['text_corrected'].fillna("")

# Split train/val (80/20)

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['level_black_humour']
)

# Tokenizzazione

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(
    list(train_df['text_corrected']),
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors='np'
)

val_encodings = tokenizer(
    list(val_df['text_corrected']),
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors='np'
)

# One-hot label

train_labels = to_categorical(train_df['level_black_humour'], num_classes=3)
val_labels = to_categorical(val_df['level_black_humour'], num_classes=3)

# Modello BERT

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

loss = CategoricalCrossentropy(from_logits=True)

model.compile(
    optimizer='adam',  # usa adam con parametri di default
    loss=loss,
    metrics=['accuracy']
)

# Callbacks

save_dir = '/content/drive/MyDrive/sarcasm_project_black'
os.makedirs(save_dir, exist_ok=True)
checkpoint_path = os.path.join(save_dir, 'best_model_black.weights.h5')

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, save_weights_only=True)

# Training

history_black = model.fit(
    {'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask']},
    train_labels,
    validation_data=(
        {'input_ids': val_encodings['input_ids'], 'attention_mask': val_encodings['attention_mask']},
        val_labels
    ),
    epochs=5,
    batch_size=32,
    callbacks=[early_stopping, checkpoint]
)

# Salva modello completo

model.save_pretrained(os.path.join(save_dir, 'bert_model_black'))
print("Training completato e modello salvato!")

from transformers import TFBertForSequenceClassification

model_black = TFBertForSequenceClassification.from_pretrained('/content/drive/MyDrive/sarcasm_project_black/bert_model_black')

import matplotlib.pyplot as plt

# Accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_black.history['accuracy'], label='Train Accuracy')
plt.plot(history_black.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over epochs')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history_black.history['loss'], label='Train Loss')
plt.plot(history_black.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over epochs')
plt.legend()

plt.tight_layout()
plt.show()

# Definisci la mappatura delle etichette sarcasmo
label_mapping = {
    'general': 2,
    'very_twisted': 1,
    'twisted_meaning': 1,
    'not_sarcastic': 0
}

# Applica la mappatura sull'intero dataframe
df['sarcasm_level'] = df['sarcasm'].map(label_mapping)
df = df.dropna(subset=['sarcasm_level'])
df['sarcasm_level'] = df['sarcasm_level'].astype(int)
print(df['sarcasm_level'].head(10))

val_df['sarcasm_level'] = val_df['sarcasm'].map(label_mapping)
print(val_df['sarcasm_level'].head(10))
print(val_df.shape)

true_sarc = val_df['sarcasm_level'].to_numpy()
true_black = val_df['level_black_humour'].to_numpy()
print(true_sarc)
print (true_black)

val_encodings_sarc = tokenizer(
    val_df['text_corrected'].tolist(), padding=True, truncation=True, max_length=128, return_tensors='tf'
)

val_encodings_black = tokenizer_black(
    val_df['text_corrected'].tolist(), padding=True, truncation=True, max_length=128, return_tensors='tf'
)

val_preds_sarc = model_sarc.predict(val_encodings_sarc)
pred_labels_sarc = val_preds_sarc.logits.argmax(axis=1)

val_preds_black = model_black.predict(val_encodings_black)
pred_labels_black = val_preds_black.logits.argmax(axis=1)

val_df = val_df.copy()
val_df["pred_sarcasm"] = pred_labels_sarc
val_df["pred_blackhumour"] = pred_labels_black
val_df["difference"] = val_df["pred_sarcasm"] - val_df["pred_blackhumour"]
more_sarcastic = val_df[val_df["pred_sarcasm"] > val_df["pred_blackhumour"]]
more_black = val_df[val_df["pred_blackhumour"] > val_df["pred_sarcasm"]]
same_level = val_df[val_df["pred_sarcasm"] == val_df["pred_blackhumour"]]
print(more_sarcastic.value_counts())
print(more_sarcastic['sarcasm_level'].value_counts())
print(more_sarcastic['level_black_humour'].value_counts())

from sklearn.metrics import accuracy_score, classification_report

print("Accuracy sarcasmo:", accuracy_score(true_sarc, pred_labels_sarc))
print("Accuracy black humour:", accuracy_score(true_black, pred_labels_black))

# Intallazione di SHAP
!pip install shap

from transformers import pipeline
import shap
shap.initjs()

pipe = pipeline(
    "text-classification",
    model="/content/drive/MyDrive/sarcasm_project_black/bert_model_black",  # path al modello salvato
    tokenizer="bert-base-uncased",  # tokenizer Hugging Face compatibile
    return_all_scores=True  # per ottenere le probabilità su tutte le classi
)

# Crea il masker e l'explainer
explainer = shap.Explainer(pipe, shap.maskers.Text())

# Campione di testi
sample_texts = val_df['text_corrected'].tolist()[:20]

# Calcolo SHAP
shap_values = explainer(sample_texts)

# Visualizzazione
shap.plots.text(shap_values[17])
