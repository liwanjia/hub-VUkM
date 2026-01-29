import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader

dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40

class CharLSTMDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# --- NEW LSTM Model Class ---
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers , bidirectional, dropout ):
        super(RNNClassifier, self).__init__()
        """
            dropout: Dropout概率
            bidirectional: 是否双向
            rnn_type: RNN类型，可选 'rnn', 'lstm', 'gru'
        """
        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 随机编码的过程， 可训练的
        self.rnn = nn.RNN(input_size=embedding_dim,
                          hidden_size = hidden_dim,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0,
                          nonlinearity='relu')  # 循环层

        # 全连接层
        rnn_output_size = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(rnn_output_size, output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 保存参数
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = 'rnn'

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        # 嵌入层
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        if self.embedding.padding_idx is not None:
            nn.init.zeros_(self.embedding.weight[self.embedding.padding_idx])

        # RNN层
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                if 'ih' in name:  # 输入到隐藏
                    nn.init.xavier_uniform_(param)
                elif 'hh' in name:  # 隐藏到隐藏
                    nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        # 全连接层
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, input_ids, lengths=None):
        batch_size = input_ids.size(0)

        # 1. 嵌入层
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(embedded)

        # 2. 如果没有提供lengths，假设没有填充
        if lengths is None:
            lengths = torch.full((batch_size,), input_ids.size(1),
                                 dtype=torch.long, device=input_ids.device)

        # 确保lengths是1D CPU张量
        if lengths.dim() > 1:
            lengths = lengths.flatten()
        lengths = lengths.cpu()

        # 3. 打包序列
        packed_embedded = pack_padded_sequence(
            embedded,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, hidden = self.rnn(packed_embedded)

        if self.bidirectional:
            # 双向RNN
            hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
            hidden_forward = hidden[-1, 0, :, :]
            hidden_backward = hidden[-1, 1, :, :]
            hidden_concat = torch.cat([hidden_forward, hidden_backward], dim=1)
        else:
            # 单向RNN
            hidden_concat = hidden[-1, :, :]

         #  Dropout
        hidden_concat = self.dropout(hidden_concat)

        # 分类
        out = self.fc(hidden_concat)
        return out

# --- Training and Prediction ---
lstm_dataset = CharLSTMDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(lstm_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, 2, True, 1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 4
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

def classify_text_lstm(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text_lstm(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_lstm(new_text_2, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
