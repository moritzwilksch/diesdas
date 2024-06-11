# %%
import polars as pl

# %%
df = pl.DataFrame(
    {
        "a": [1, 2, 3, None],
        "b": ["1", "2", "3", None],
    }
).lazy()

md = {1: 11, 2: 22}
# md = {1: "11", 2: "22"}
ret_dtype = pl.Series(md.values()).dtype
df.select(
    # pl.col("b").map_dict(md).fill_null(99).cast(ret_dtype).alias("test")
    pl.col("b").replace(md).fill_null(99).alias("test")
).collect()
