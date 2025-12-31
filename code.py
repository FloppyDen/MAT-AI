import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import random

# Синтетические данные (для демонстрации)
texts = [
    "Привет, как дела?",  # чистый (0)
    "Ты полный идиот!",  # мат (1)
    "Приветт, каак делаа?",  # неточности (2)
    "Ты хуй, идиотт!",  # оба (3)
] * 250  # 1000 примеров

labels = [0, 1, 2, 3] * 250

# Токенизатор
tokenizer = get_tokenizer('basic_english')  # Простой, для русского адаптировать

# Построение словаря
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text.lower())

vocab = build_vocab_from_iterator(yield_tokens(texts), specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])

# Датасет
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=50):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = tokenizer(text.lower())[:self.max_len]
        indices = [self.vocab[token] for token in tokens]
        length = len(indices)
        return torch.tensor(indices), torch.tensor(label), length

def collate_batch(batch):
    texts, labels, lengths = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=vocab["<pad>"])
    return texts_padded, torch.tensor(labels), torch.tensor(lengths)

# Разделение данных
train_size = int(0.8 * len(texts))
train_texts, val_texts = texts[:train_size], texts[train_size:]
train_labels, val_labels = labels[:train_size], labels[train_size:]

train_dataset = TextDataset(train_texts, train_labels, vocab)
val_dataset = TextDataset(val_texts, val_labels, vocab)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_batch)

# Модель
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_classes=4, dropout=0.3):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab["<pad>"])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 из-за bidirectional

    def forward(self, x, lengths):
        embed = self.embedding(x)
        packed = pack_padded_sequence(embed, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, (hidden, _) = self.lstm(packed)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)  # Для bidirectional
        drop = self.dropout(hidden)
        out = self.fc(drop)
        return out

# Инициализация
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextClassifier(len(vocab), embed_dim=100, hidden_dim=128, num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for texts, labels, lengths in loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for texts, labels, lengths in loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

# Цикл обучения
epochs = 10
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

# Тестирование (пример)
test_text = "Ты полный хуй!"
tokens = tokenizer(test_text.lower())
indices = torch.tensor([vocab[token] for token in tokens]).unsqueeze(0).to(device)
length = torch.tensor([len(tokens)])
model.eval()
with torch.no_grad():
    output = model(indices, length)
    pred = output.argmax(dim=1).item()
print(f"Предсказание: {pred}")  # 1 или 3