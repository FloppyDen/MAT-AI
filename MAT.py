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

# ==================== 1. Проверка и установка GPU ====================
print("Проверка доступности GPU...")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU найден: {torch.cuda.get_device_name(0)}")
    print(f"Количество GPU: {torch.cuda.device_count()}")
else:
    device = torch.device("cpu")
    print("GPU не найден. Будет использоваться CPU (обучение будет медленнее).")

# Очистка кэша GPU (полезно при повторных запусках)
if device.type == "cuda":
    torch.cuda.empty_cache()

# ==================== 2. Загрузка датасета из CSV ====================
print("\nЗагрузка датасета labeled.csv...")
df = pd.read_csv('labeled.csv')

print(f"Всего комментариев: {len(df)}")
print(df['toxic'].value_counts())

# Функция добавления опечаток
def add_typos(text, prob=0.4):
    words = list(tokenize(text))
    new_words = []
    for word in words:
        if random.random() < prob and len(word.text) > 4:
            pos = random.randint(1, len(word.text)-1)
            if random.random() < 0.5:
                typo = word.text[:pos] + word.text[pos] + word.text[pos:]  # дублирование
            else:
                typo = word.text[:pos] + word.text[pos+1:]  # удаление
            new_words.append(typo)
        else:
            new_words.append(word.text)
    return ' '.join(new_words)

# Формирование датасета с 3 классами
texts = []
labels = []

# Класс 0: чистый
clean_df = df[df['toxic'] == 0].sample(n=3000, random_state=42)
texts.extend(clean_df['comment'].tolist())
labels.extend([0] * len(clean_df))

# Класс 1: мат/токсичность
toxic_df = df[df['toxic'] == 1].sample(n=3000, random_state=42)
texts.extend(toxic_df['comment'].tolist())
labels.extend([1] * len(toxic_df))

# Класс 2: с опечатками (на основе чистых)
typo_texts = [add_typos(text, prob=0.4) for text in clean_df['comment'].tolist()[:3000]]
texts.extend(typo_texts)
labels.extend([2] * len(typo_texts))

print(f"Итоговый размер датасета: {len(texts)}")
print(f"Распределение классов: {Counter(labels)}")

# ==================== 3. Токенизация и словарь ====================
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

def text_to_indices(text, vocab, max_len=100):
    tokens = russian_tokenizer(text)[:max_len]
    return [vocab.get(token, vocab['<unk>']) for token in tokens]

# ==================== 4. Dataset ====================
class ToxicDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
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
    lengths = torch.tensor([len(t) for t in texts])  # уже тензор
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=vocab['<pad>'])
    return texts_padded, torch.tensor(labels), lengths

# Разделение и DataLoader
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

train_dataset = ToxicDataset(train_texts, train_labels, vocab)
val_dataset = ToxicDataset(val_texts, val_labels, vocab)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)

# ==================== Улучшенная модель и оптимизатор ====================
MODEL_PATH = "best_mat_detector.pth"  # Файл для сохранения лучшей модели

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, num_classes=3, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2,  # 2 слоя вместо 1
                            batch_first=True, bidirectional=True, dropout=0.5)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)  # Нормализация для стабильности

    def forward(self, x, lengths):
        embed = self.embedding(x)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embed, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # bidirectional
        hidden = self.layer_norm(hidden)
        out = self.dropout(hidden)
        out = self.fc(out)
        return out

# Создание модели
model = TextClassifier(len(vocab), embed_dim=256, hidden_dim=256, num_classes=3).to(device)

# Улучшенный оптимизатор с weight decay и scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # AdamW лучше Adam

# Scheduler для снижения lr при застое
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
# ==================== Загрузка модели, если она уже обучена ====================
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Загружена сохранённая модель из {MODEL_PATH}")
    print("Обучение пропущено — используем готовую модель.\n")

    # Быстрая проверка accuracy на валидации
    quick_acc = evaluate()
    print(f"Текущая точность загруженной модели: {quick_acc:.4f}\n")
else:
    print("Сохранённой модели не найдено — начинаем обучение с нуля.\n")
# Запуск обучения
epochs = 20  # Можно больше — early stopping остановит раньше
patience = 5  # Сколько эпох ждать улучшения
no_improve = 0

print("\nНачало обучения...\n")

for epoch in range(epochs):
    train_loss = train_epoch()
    val_acc = evaluate()

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Accuracy: {val_acc:.4f}")

    # Сохранение лучшей модели
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  >>> Новая лучшая модель сохранена! Accuracy: {val_acc:.4f}")
        no_improve = 0
    else:
        no_improve += 1

    scheduler.step(val_acc)  # Уменьшаем lr при отсутствии улучшения

    # Early stopping
    if no_improve >= patience:
        print(f"\nEarly stopping на эпохе {epoch+1}. Лучшая accuracy: {best_val_acc:.4f}")
        break

print(f"\nОбучение завершено. Лучшая accuracy: {best_val_acc:.4f}")

# ==================== 7. Тестирование ====================
model.eval()
test_examples = [
    "Привет, как дела?",
    "Ты дебил конченый!",
    "Приветт, каак у теббя делла?",
]

print("\nТестирование модели:")
with torch.no_grad():
    for text in test_examples:
        indices = torch.tensor([text_to_indices(text, vocab)]).to(device)
        lengths = torch.tensor([len(indices[0])]).to(device)
        output = model(indices, lengths)
        pred = output.argmax(dim=1).item()
        classes = ["чистый", "мат/токсичность", "опечатки"]
        print(f'"{text}" → {pred} ({classes[pred]})')

# ==================== 8. Интерактивное окно для проверки текста ====================
def launch_interactive_checker():
    model.eval()  # Убеждаемся, что модель в режиме оценки

    def check_text():
        input_text = text_input.get("1.0", tk.END).strip()
        if not input_text:
            result_label.config(text="Введите текст для проверки!", foreground="orange")
            return
        
        # Преобразование введённого текста в тензор
        indices = text_to_indices(input_text, vocab, max_len=100)
        input_tensor = torch.tensor([indices]).to(device)
        length_tensor = torch.tensor([len(indices)]).to(device)
        
        with torch.no_grad():
            output = model(input_tensor, length_tensor)
            pred = output.argmax(dim=1).item()
        
        classes = ["чистый", "мат / токсичность", "опечатки / ошибки"]
        confidence = torch.softmax(output, dim=1).max().item() * 100
        
        result_text = f"Результат: {classes[pred]}\nУверенность: {confidence:.1f}%"
        result_label.config(
            text=result_text,
            foreground="green" if pred == 0 else "red" if pred == 1 else "orange"
        )

    # Создание окна
    root = tk.Tk()
    root.title("Проверка текста на токсичность и ошибки")
    root.geometry("600x500")
    root.resizable(True, True)
    
    # Заголовок
    title_label = ttk.Label(root, text="Детектор мата и опечаток", font=("Arial", 16, "bold"))
    title_label.pack(pady=10)
    
    # Поле ввода
    ttk.Label(root, text="Введите текст для проверки:").pack(anchor="w", padx=20)
    text_input = scrolledtext.ScrolledText(root, height=10, wrap=tk.WORD, font=("Arial", 12))
    text_input.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
    
    # Кнопка проверки
    check_button = ttk.Button(root, text="Проверить", command=check_text)
    check_button.pack(pady=10)
    
    # Метка с результатом
    result_label = ttk.Label(root, text="Результат появится здесь", font=("Arial", 14), justify="center")
    result_label.pack(pady=20)
    
    # Примеры для быстрого теста
    examples_frame = ttk.LabelFrame(root, text="Быстрые примеры")
    examples_frame.pack(padx=20, pady=10, fill=tk.X)
    
    def insert_example(example_text):
        text_input.delete("1.0", tk.END)
        text_input.insert("1.0", example_text)
    
    ttk.Button(examples_frame, text="Чистый текст", 
               command=lambda: insert_example("Привет! Как дела? Всё отлично.")).grid(row=0, column=0, padx=5, pady=5)
    ttk.Button(examples_frame, text="Мат", 
               command=lambda: insert_example("Ты чё, дебил конченый?")).grid(row=0, column=1, padx=5, pady=5)
    ttk.Button(examples_frame, text="Опечатки", 
               command=lambda: insert_example("Приветт, каак у теббя делла?")).grid(row=0, column=2, padx=5, pady=5)
    
    root.mainloop()

# ==================== Запуск интерактивного окна после обучения ====================
print("\nОбучение завершено!")
print("Запуск интерактивного проверщика текста...")

launch_interactive_checker()