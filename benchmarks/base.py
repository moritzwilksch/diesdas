# %%
import duckdb
import numpy as np
import polars as pl
import time

rng = np.random.default_rng(42)


class Experiment: ...


# %%
N = 10_000_000
data = []
for _ in range(N):
    size = rng.integers(0, 10)
    l = rng.integers(0, 500, size=size)
    data.append(l)

# %%
df = pl.DataFrame({"a": data}).lazy()

# %%
pl.Config(verbose=True)
tic = time.perf_counter()
for _ in range(10):
    df.select(
        pl.col("a").list.count_matches(n).cast(pl.UInt8).name.suffix(f"_{n}")
        for n in range(500)
    ).collect()
tac = time.perf_counter()
print(f"Took {(tac - tic) * 1000 / 10:.0f}ms/iter")

# %%
rel = duckdb.from_arrow(df.collect().to_arrow())
duckdb.register("rel", df.collect().to_arrow())
tic = time.perf_counter()
for _ in range(10):
    res = rel.select(
        # ",".join([f"list_contains(a, {n}) as a_{n}" for n in range(500)])
        ",".join([f"list_filter(a, e -> e = {n}).len() as a_{n}" for n in range(500)])
        # ",".join([f"list_transform(a, e -> (e = {n})::uint8).list_sum() as a_{n}" for n in range(500)])
        # ",".join([f"[(e = {n})::int for e in a].list_sum()" for n in range(500)])
    ).execute()
tac = time.perf_counter()
print(f"Took {(tac - tic) * 1000 / 10:.0f}ms/iter")

# %%
stmt = ",".join(f"list_filter(a, e->e={n}).len()::int8 as a_{n}" for n in range(500))
res = duckdb.execute(f"""
explain analyze SELECT
{stmt}
from rel
""")


# %%
stmt = ",".join(f"a = {n} as a_{n}" for n in range(500))
stmt2 = ",".join(f"sum(a_{n}::uint8)" for n in range(500))
duckdb.sql(f"""
with cte as (
select idx, {stmt}
from (select row_number() over () as idx, unnest(a) as a from rel)
)

select any_value(idx), {stmt2}
from cte
""")


# %%
df2 = pl.DataFrame({"a": [[1, 2, 3], [], [3, 3, 4]]})
duckdb.sql("""
select [e for e in a if e = 3].len()
from df2
""")
# %%


rel.select(
    ",".join(
        [
            f"case when a then list_reduce(a, (x, y) -> x + (case when y = {n} then 1 else 0 end)) else 0 end"
            for n in range(5)
        ]
    )
)

# %%
rel.select(
    ",".join(
        [
            f"list_transform(a, e -> case when e = {n} then 1 else 0 end).list_sum() as a_{n}"
            for n in range(500)
        ]
    )
).execute()

# %%
rel.select("a, list_count(a)")
# %%
res, profile = (
    df.with_row_index()
    .explode("a")
    .group_by("index")
    .agg(pl.col("a").eq(n).sum().name.suffix(f"_{n}") for n in range(500))
    .profile()
)

# %%
profile.with_columns(dur=pl.col("end") - pl.col("start")).with_columns(
    rel=pl.col("dur") / pl.col("dur").sum()
).sort("dur")
