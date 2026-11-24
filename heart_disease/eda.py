# %%
import pandas as pd

# %%
train_df = pd.read_csv("data/train.csv")
TARGET = "History of HeartDisease or Attack"
# %%
print(train_df.shape)
print("count :", train_df[TARGET].value_counts(dropna = False))

# %%
missing = train_df[TARGET].isna()
print(missing)
print(missing.sum())
print(missing.mean())

yes_diabete = train_df[missing]
no_diabete = train_df[~missing]

print(yes_diabete.head(5))
print(no_diabete.head(5))

# %%
train_df.groupby(missing)[["Age", "Body Mass Index"]].mean()

# %%
train_df.groupby(missing)["General Health"].value_counts(normalize=True).unstack(fill_value=0)

# %%
y_true = train_df[TARGET]
y_pred = ["No"] * len(y_true)

