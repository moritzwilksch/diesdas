# %%
import polars as pl

df = pl.DataFrame({c: list(range(10)) for c in "abcdefghijklmno"})
df

# %%
df.with_columns((pl.lit(1) + pl.all()).name.suffix("_"))

# %%
df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
df.with_columns(1 + pl.col("a"))
# ComputeError: the name: 'literal' passed to `LazyFrame.with_columns` is duplicate

# %%
df.with_columns(pl.lit(1) + pl.all())
# ComputeError: the name: 'literal' passed to `LazyFrame.with_columns` is duplicate
