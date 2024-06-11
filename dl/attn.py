# %%
import torch

from torch import nn, outer


class Attn(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.w_k = nn.Linear(d_model, d_k)
        self.w_q = nn.Linear(d_model, d_k)
        self.w_v = nn.Linear(d_model, d_v)

    def forward(self, X):
        Q = self.w_q(X)
        K = self.w_k(X)
        V = self.w_v(X)

        attn = Q @ K.transpose(0, 1) / self.d_k**0.5
        attn_softmax = torch.softmax(attn, dim=1)

        return attn_softmax @ V


class MultiHeadAttn(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model

        self.heads = [Attn(d_model, d_k, d_v) for _ in range(n_head)]
        self.w_o = nn.Linear(n_head * d_v, d_model)

    def forward(self, X):
        out_heads = [h.forward(X) for h in self.heads]
        concat = torch.concat(out_heads, dim=-1)
        return self.w_o(concat)


attn = MultiHeadAttn(128, 64, 32, 2)

X = torch.randn((9, 128))
attn(X).shape


# %%
import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader

train_iter = iter(AG_NEWS(split="train"))
tokenizer = get_tokenizer("basic_english")
train_iter = AG_NEWS(split="train")


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    # offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    # text_list = torch.nested.nested_tensor(text_list)
    text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=False)
    return label_list.to(device), text_list.to(device)


train_iter = AG_NEWS(split="train")
dataloader = DataLoader(
    train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch
)


# %%


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(len(vocab), embedding_dim=128)
        self.attn = MultiHeadAttn(128, 64, 64, 8)
        self.ff = nn.Linear(128, 20)

    def forward(self, x):
        x = self.embed(x)
        x = self.attn(x)
        x = self.ff(x)
        return torch.softmax(x)


model = MyModel()

for epoch in range(10):
    for batch in dataloader:
        y, x = batch

        out = model(x)
        breakpoint()

