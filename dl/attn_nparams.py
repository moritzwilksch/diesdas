# %%
import numpy as np


def softmax(x, axis=0):
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


print(softmax(np.array([1, 3, 2])))
print(softmax(np.array([[1, 3, 2], [1, 3, 2]]), axis=1))
print(softmax(np.array([[1, 3, 2], [1, 3, 2]]), axis=0))


# %%
def attn_forward(
    X,
    d_k: int,
    d_v: int,
    num_heads: int,
):
    seq_len, d_model = X.shape
    head_dim = d_v // num_heads

    W_Q = np.random.normal(size=(d_model, d_k))
    W_K = np.random.normal(size=(d_model, d_k))
    W_V = np.random.normal(size=(d_model, head_dim * num_heads))

    Q = X @ W_Q  # s, d_k
    K = X @ W_K  # s, d_k
    V = X @ W_V  # d_model, head_dim * num_heads

    attn = softmax(Q @ K.T / np.sqrt(d_k), axis=1) @ V

    return attn


attn_forward(
    np.random.random(size=(2, 128)),
    d_k=32,
    d_v=32,
    num_heads=2,
).shape


# %%
def attn_parameter_count(
    d_model: int,
    d_k: int,
    d_v: int,
    num_heads: int,
):
    assert d_v % num_heads == 0
    return dict(
        W_Q=d_model * d_k,
        W_K=d_model * d_k,
        W_V=d_model * d_v // num_heads,
        W_O=d_v // num_heads * d_model,
    )


configs = [
    {"d_model": 128, "d_k": 32, "d_v": 32, "num_heads": 1},
    {"d_model": 128, "d_k": 32, "d_v": 16, "num_heads": 2},
    {"d_model": 128, "d_k": 32, "d_v": 8, "num_heads": 4},
]

for config in configs:
    param_count = attn_parameter_count(**config)
    print(config)
    print(param_count)
    print(f"{sum(param_count.values()):,.0f}")
