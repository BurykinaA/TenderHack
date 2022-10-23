import pandas as pd
import json

df = pd.read_excel("/home/vadim/K/hack/dataset/cte.xlsx")
df["label"] = df["Код КПГЗ"].astype(str).apply(lambda x: x.split('.')[0] + "." + x.split('.')[1] if len(x.split('.')) != 1 else x.split('.')[0])

with open('codes.txt', 'r') as file:
    lines = file.readlines()

lines = [l.strip() for l in lines if l.startswith('            ') and l[12] != ' ']

codes = {}
for l in lines:
    codes[l[:5]] = l[7:]

decoding = {}
m = {}
for i, u in enumerate(df["label"].unique()):
    m[u] = i 
    if u == "nan":
        decoding[i] = None
    else:
        decoding[i] = codes[u]

print(list(df["label"].values).count('nan'))
df["label"] = df["label"].map(m)

df[["Название СТЕ", "label"]].to_csv("/home/vadim/K/hack/dataset/train_bert.csv")
print(df["label"].nunique())
print(dict(df["label"].value_counts()))

with open('codes.json', 'w') as file:
    json.dump(decoding, file)