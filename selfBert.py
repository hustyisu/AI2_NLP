import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载IMDB数据集
# 认为已经存入DataFrame格式的data中

# 划分数据集
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 构建词汇表
class Vocabulary:
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.index = 0
    
    def add_word(self, word):
        if word not in self.word_to_index:
            self.word_to_index[word] = self.index
            self.index_to_word[self.index] = word
            self.index += 1
    
    def __len__(self):
        return len(self.word_to_index)

vocab = Vocabulary()
for text in train_data['review']:
    for word in text.split():
        vocab.add_word(word)

# 数据预处理和准备
def preprocess_text(text, vocab, max_length):
    tokens = text.split()[:max_length]
    input_ids = [vocab.word_to_index.get(token, vocab.word_to_index['<unk>']) for token in tokens]
    input_ids = input_ids + [vocab.word_to_index['<pad>']] * (max_length - len(input_ids))
    return input_ids

max_length = 128
vocab.add_word('<pad>')
vocab.add_word('<unk>')

train_input_ids = [preprocess_text(text, vocab, max_length) for text in train_data['review']]
test_input_ids = [preprocess_text(text, vocab, max_length) for text in test_data['review']]

train_labels = np.array(train_data['label'])
test_labels = np.array(test_data['label'])

# 转换为PyTorch的Tensor
train_input_ids = torch.tensor(train_input_ids)
train_labels = torch.tensor(train_labels)

test_input_ids = torch.tensor(test_input_ids)
test_labels = torch.tensor(test_labels)

# 构建简单的BERT模型
class BERT(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_classes):
        super(BERT, self).__init__()
        # 使用embedding
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        pooled = torch.mean(embedded, dim=1)  # 简单平均池化
        logits = self.fc(pooled)
        return logits

# 初始化BERT模型
vocab_size = len(vocab)
hidden_size = 256
num_classes = 2
bert_model = BERT(vocab_size, hidden_size, num_classes)

# 设置训练参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(bert_model.parameters(), lr=2e-5)

# 训练模型
num_epochs = 3
batch_size = 32

for epoch in range(num_epochs):
    bert_model.train()
    for i in range(0, len(train_input_ids), batch_size):
        inputs = train_input_ids[i:i+batch_size].to(device)
        labels = train_labels[i:i+batch_size].to(device)
        
        optimizer.zero_grad()
        logits = bert_model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        if (i // batch_size) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Batch {i//batch_size}, Loss: {loss.item()}')

# 模型评估
bert_model.eval()
with torch.no_grad():
    predictions = []
    for i in range(0, len(test_input_ids), batch_size):
        inputs = test_input_ids[i:i+batch_size].to(device)
        
        logits = bert_model(inputs)
        predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(predicted_labels)

accuracy = accuracy_score(test_labels, predictions)
print(f'Accuracy on test set: {accuracy:.4f}')
