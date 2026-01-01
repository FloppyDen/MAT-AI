import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import tkinter as tk
from tkinter import ttk, scrolledtext
import os
import pandas as pd
import random
from razdel import tokenize
from collections import Counter
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("GPU не обнаружен. Обучение будет на CPU.")


print("\nЗагрузка датасета labeled.csv...")
df = pd.read_csv('labeled.csv',sep=',', encoding='utf-8')

print(f"Всего комментариев: {len(df)}")
print(df['toxic'].value_counts())

print("Загрузка датасета orfo_check.csv для ошибок/неточностей...")
typos_df = pd.read_csv('orfo_check.csv', sep=';', encoding='utf-8')


def add_typos(text, prob=0.7):
    words = list(tokenize(text))
    new_words = []
    for word in words:
        if random.random() < prob and len(word.text) > 4:
            pos = random.randint(1, len(word.text)-1)
            if random.random() < 0.4:
                typo = word.text[:pos] + word.text[pos] + word.text[pos:] 
            else:
                typo = word.text[:pos] + word.text[pos+1:]  
            new_words.append(typo)
        else:
            new_words.append(word.text)
    return ' '.join(new_words)


texts = []
labels = []





# Класс чистый
clean_df = df[df['toxic'] == 0].sample(n=4000, random_state=42)
texts.extend(clean_df['comment'].tolist())
labels.extend([0] * len(clean_df))
texts.extend(typos_df['CORRECT'].dropna().unique().tolist()[:1500])
labels.extend([0] * len(typos_df['CORRECT'].dropna().unique().tolist()[:1500]))

# Класс мат
toxic_df = df[df['toxic'] == 1].sample(n=3000, random_state=42)
texts.extend(toxic_df['comment'].tolist())
labels.extend([1] * len(toxic_df))

# Класс ошибки
typo_base = clean_df['comment'].tolist() + toxic_df['comment'].tolist()
typo_texts = [add_typos(text, prob=0.7) for text in clean_df['comment'].tolist()[:4000]]
texts.extend(typo_texts)
labels.extend([2] * len(typo_texts))


mistake_texts = typos_df['MISTAKE'].dropna().astype(str).tolist()
texts.extend(mistake_texts)
labels.extend([2] * len(mistake_texts))

print(f"Итоговый размер датасета: {len(texts)}")
print(f"Распределение классов: {Counter(labels)}")
print(f"Добавлено {len(typos_df['MISTAKE'].dropna())} примеров с ошибками в класс 2.")
print(f"Добавлено {len(typos_df['CORRECT'].dropna().unique())} чистых слов в класс 0.")

def russian_tokenizer(text):
    return [token.text.lower() for token in tokenize(text)]

def build_vocab(texts, min_freq=2):
    counter = Counter()
    for text in texts:
        counter.update(russian_tokenizer(text))
    
    vocab = {'<pad>': 0, '<unk>': 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

vocab = build_vocab(texts)
print(f"Размер словаря: {len(vocab)}")

def text_to_indices(text, vocab, max_len=400):
    tokens = russian_tokenizer(text)[:max_len]
    return [vocab.get(token, vocab['<unk>']) for token in tokens]


class ToxicDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=400):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        indices = text_to_indices(self.texts[idx], self.vocab, self.max_len)
        return torch.tensor(indices), torch.tensor(self.labels[idx])

def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(t) for t in texts])  
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=vocab['<pad>'])
    return texts_padded, torch.tensor(labels), lengths


train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

train_dataset = ToxicDataset(train_texts, train_labels, vocab)
val_dataset = ToxicDataset(val_texts, val_labels, vocab)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)


MODEL_PATH = "best_mat_detector.pth"  # Файл для сохранения лучшей модели

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, num_classes=3, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=4,  # 2 слоя вместо 1
                            batch_first=True, bidirectional=True, dropout=0.5)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)  

    def forward(self, x, lengths):
        embed = self.embedding(x)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embed, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  
        hidden = self.layer_norm(hidden)
        out = self.dropout(hidden)
        out = self.fc(out)
        return out


model = TextClassifier(len(vocab), embed_dim=256, hidden_dim=256, num_classes=3).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # AdamW лучше Adam


scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,)

best_val_acc = 0.0

def train_epoch():
    model.train()
    total_loss = 0
    for texts, labels, lengths in train_loader:
        texts = texts.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(texts, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Защита от взрыва градиентов
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels, lengths in val_loader:
            texts = texts.to(device)
            labels = labels.to(device)
            outputs = model(texts, lengths)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total
if os.path.exists(MODEL_PATH):
    pretrained_dict = torch.load(MODEL_PATH, map_location=device)
    model_dict = model.state_dict()
    
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    print(f"Загружена модель из {MODEL_PATH} (частично: {len(pretrained_dict)}/{len(model_dict)} слоёв)")
    print("Новые слова в словаре инициализированы случайно.")
else:
    print("Сохранённой модели не найдено — обучение с нуля.")

epochs = 40 
patience = 7  
no_improve = 0

print("\nНачало обучения...\n")

for epoch in range(epochs):
    train_loss = train_epoch()
    val_acc = evaluate()

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  >>> Новая лучшая модель сохранена! Accuracy: {val_acc:.4f}")
        no_improve = 0
    else:
        no_improve += 1

    scheduler.step(val_acc) 

    
    if no_improve >= patience:
        print(f"\nEarly stopping на эпохе {epoch+1}. Лучшая accuracy: {best_val_acc:.4f}")
        break

print(f"\nОбучение завершено. Лучшая accuracy: {best_val_acc:.4f}")




def launch_interactive_checker():
    model.eval()  

    def check_text():
        input_text = text_input.get("1.0", tk.END).strip()
        if not input_text:
            result_label.config(text="Введите текст для проверки!", foreground="orange")
            return
        

        indices = text_to_indices(input_text, vocab, max_len=100)
        input_tensor = torch.tensor([indices]).to(device)
        length_tensor = torch.tensor([len(indices)]).to(device)
        
        with torch.no_grad():
            output = model(input_tensor, length_tensor)
            pred = output.argmax(dim=1).item()
        
        classes = ["чистый", "ненормативная лексикf", "опечатки/ошибки"]
        confidence = torch.softmax(output, dim=1).max().item() * 100
        
        result_text = f"Результат: {classes[pred]}\nУверенность: {confidence:.1f}%"
        result_label.config(
            text=result_text,
            foreground="green" if pred == 0 else "red" if pred == 1 else "orange"
        )

 
    root = tk.Tk()
    root.title("Применение нейронных сетей для обнаружения некорректной и ненормативной лексики")
    root.geometry("600x500")
    root.resizable(True, True)
    title_label = ttk.Label(root, text="Применение нейронных сетей для обнаружения некорректной и ненормативной лексики", font=("Arial", 16, "bold"))
    ttk.Label(root, text="Введите текст для проверки:").pack(anchor="w", padx=20)
    text_input = scrolledtext.ScrolledText(root, height=10, wrap=tk.WORD, font=("Arial", 12))
    text_input.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
    
    check_button = ttk.Button(root, text="Проверить", command=check_text)
    check_button.pack(pady=10)
    
    result_label = ttk.Label(root, text="Результат появится здесь", font=("Arial", 14), justify="center")
    result_label.pack(pady=20)
    
    root.mainloop()

print("\nОбучение завершено!")
print("Запуск интерактивного проверщика текста")

launch_interactive_checker()