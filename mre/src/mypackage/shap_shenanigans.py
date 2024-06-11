# %%
import numpy as np
import lightgbm as lgbm
import seaborn as sns

# %%
data = sns.load_dataset("iris").query('species != "versicolor"')
X = data.drop(columns="species")
y = data["species"]
# %%
model = lgbm.LGBMClassifier()
model.fit(X, y)

# %%
preds = model.predict_proba(X)[:, 1]
scores = model.predict(X, raw_score=True)
shap = model.predict(X, pred_contrib=True)


# %%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


assert np.allclose(sigmoid(scores), preds)  # holds

# %%
assert np.allclose(shap.sum(axis=1), scores)  # holds

# %%
shap2 = model.predict(X.head(), pred_contrib=True)

assert np.allclose(shap2, shap[:5])  # holds
