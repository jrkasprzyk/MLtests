from annoy import AnnoyIndex
import pandas as pd

df = pd.read_csv("condensed_policies_all.csv")
num_dec_vars = len(df.columns)-2

t = AnnoyIndex(num_dec_vars, "manhattan")

for index, row in df.drop(columns=["Experiment", "Policy"]).iterrows():
    t.add_item(index, row)

t.build(10)
t.save("analyze_solutions.ann")

for index, row in df.iterrows():
    neighbors = t.get_nns_by_item(index, 4)
    df.loc[index, "Nbr1_ix"] = neighbors[1]
    df.loc[index, "Nbr1_exp"] = df.loc[neighbors[1], "Experiment"]
    df.loc[index, "Nbr1_pol"] = df.loc[neighbors[1], "Policy"]
    df.loc[index, "Nbr2_ix"] = neighbors[2]
    df.loc[index, "Nbr2_exp"] = df.loc[neighbors[2], "Experiment"]
    df.loc[index, "Nbr2_pol"] = df.loc[neighbors[2], "Policy"]
    df.loc[index, "Nbr3_ix"] = neighbors[3]
    df.loc[index, "Nbr3_exp"] = df.loc[neighbors[3], "Experiment"]
    df.loc[index, "Nbr3_pol"] = df.loc[neighbors[3], "Policy"]

df.to_csv("condensed_policies_all_with_neighbors.csv")

