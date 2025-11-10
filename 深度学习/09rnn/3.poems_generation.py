import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

import re

# 1. 数据预处理
def preprocess_poems(file_path):
    # 定义字的集合（去重），保存诗id化之后的列表
    char_set = set()
    poems = []
    # 1.1 逐行读取文件，保存诗的内容
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            # 数据清洗，去掉标点以及两侧空白
            line = re.sub(r"[，。？！、：]", "", line).strip()
            # 按字分割并去重，保存到set
            char_set.update(list(line))
            poems.append(list(line))
    # 1.2 构建词表
    id2word = list(char_set) + ["<UNK>"]
    word2id = { word:id for id, word in enumerate(id2word) }
    # 1.3 将诗句id化
    id_seqs = []
    for poem in poems:
        id_seq = [ word2id.get(word) for word in poem ]
        id_seqs.append(id_seq)
    return id_seqs, id2word, word2id

id_seqs, id2word, word2id = preprocess_poems("../data/poems.txt")

print(len(id_seqs))
print(len(id2word))

# 2. 构建自定义数据集
class PoetryDataset(Dataset):
    # 初始化：传入原诗的id列表，以及序列长度L
    def __init__(self, id_seqs, seq_len):
        self.seq_len = seq_len
        self.data = []  # 保存数据元组（x, y）的列表
        # 遍历所有诗
        for id_seq in id_seqs:
            # 遍历当前诗的所有字（id）
            for i in range(0, len(id_seq) - self.seq_len):
                # 以当前字id为起始，截取x和y序列
                self.data.append( (id_seq[i:i+self.seq_len], id_seq[i+1:i+1+self.seq_len]) )
    # 返回数据集的大小
    def __len__(self):
        return len(self.data)
    # 通过索引idx获取元素值
    def __getitem__(self, idx):
        x = torch.LongTensor(self.data[idx][0])
        y = torch.LongTensor(self.data[idx][1])
        return x, y

dataset = PoetryDataset(id_seqs, 24)
print(len(dataset))

# 3. 创建RNNLM模型
class PoetryRNN(nn.Module):
    # 初始化
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
        super().__init__()
        # 词嵌入层
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        # RNN层
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        # 输出层（全连接层）
        self.linear = nn.Linear(hidden_size, vocab_size)
    # 前向传播
    def forward(self, input, hx=None):
        embed = self.embed(input)
        output, hn = self.rnn(embed, hx)
        output = self.linear(output)
        return output, hn

model = PoetryRNN(vocab_size=len(id2word), embedding_dim=256, hidden_size=512, num_layers=2)

# 4. 模型训练
def train(model, dataset, lr, epoch_num, batch_size, device):
    # 4.1 初始化相关
    model.to(device)
    model.train()
    # 定义损失函数和优化器
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4.2 迭代训练过程
    for epoch in range(epoch_num):
        train_loss = 0
        # 定义数据加载器
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            # 4.2.1 前向传播
            output, _ = model(x)
            # 4.2.2 计算损失
            loss_value = loss(output.transpose(1,2), y)
            # 4.2.3 反向传播
            loss_value.backward()
            # 4.2.4 更新参数
            optimizer.step()
            # 4.2.5 梯度清零
            optimizer.zero_grad()

            train_loss += loss_value.item() * x.shape[0]

            # 打印进度条
            print(f"\rEpoch:{epoch + 1:0>2}[{'=' * int((batch_idx + 1) / len(dataloader) * 50)}]", end="")

        # 本轮训练结束，计算平均损失
        this_loss = train_loss / len(dataset)
        print(f"train loss: {this_loss:.4f}")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# 超参数
lr = 1e-3
epoch_num = 20
batch_size = 32

train(model, dataset, lr, epoch_num, batch_size, device)

# 5. 生成新诗（测试）
def generate_poem(model, id2word, word2id, start_token, line_num=4, line_length=7):
    model.eval()
    poem = []  # 记录生成结果
    current_rest_len = line_length  # 记录当前诗句的剩余字数
    # 5.1 token id化
    start_id = word2id.get(start_token, word2id["<UNK>"])
    if start_id != word2id["<UNK>"]:
        poem.append(start_token)
        current_rest_len -= 1
    # 5.2 定义输入数据
    input = torch.LongTensor( [[start_id]] ).to(device)
    # 5.3 迭代生成诗句
    with torch.no_grad():
        # 按行生成诗句
        for i in range(line_num):
            # 每行生成两句诗
            for interpunction in ["，","。\n"]:
                # 逐字生成诗句
                while current_rest_len > 0:
                    # 前向传播，生成下一个字的id
                    output, _ = model(input)
                    # 得到每个词的分类概率值
                    prob = torch.softmax(output[0,0], dim=-1)
                    # 基于概率分布，得到下一个随机的id
                    next_id = torch.multinomial(prob, num_samples=1)
                    # 将id转换成word，放入列表
                    poem.append( id2word[next_id.item()] )
                    # 更新input，长度减1
                    current_rest_len -= 1
                    input = next_id.unsqueeze(0)
                # 本句生成结束，添加标点符号
                poem.append( interpunction )
                current_rest_len = line_length

    return "".join(poem)

for i in range(10):
    print( generate_poem(model, id2word, word2id, start_token="一", line_num=4, line_length=7) )