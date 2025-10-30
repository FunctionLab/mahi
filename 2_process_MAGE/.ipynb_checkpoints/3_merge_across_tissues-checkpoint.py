import os
import pandas as pd

edge_dict = {}

top = 'top1'

print(top)

filtered_files = [f for f in os.listdir() if f.endswith(f'_filtered_{top}.dat')]

for f in filtered_files:
    tissue_name = f.replace(f'_filtered_{top}.dat', '')
    print(f"processing {f} as tissue: {tissue_name}")

    df = pd.read_csv(f, sep='\t', header=None, names=['source', 'target', 'weight'])

    df['source'] = df[['source', 'target']].min(axis=1)
    df['target'] = df[['source', 'target']].max(axis=1)
    df['edge'] = list(zip(df['source'], df['target']))
    
    for edge in df['edge']:
        if edge not in edge_dict:
            edge_dict[edge] = {}
        edge_dict[edge][tissue_name] = 1

all_edges = []
for edge, tissues in edge_dict.items():
    edge_entry = {
        'source': edge[0],
        'target': edge[1]
    }
    for f in filtered_files:
        tissue_name = f.replace(f'_filtered_{top}.dat', '')
        edge_entry[tissue_name] = tissues.get(tissue_name, 0)
    all_edges.append(edge_entry)

master_df = pd.DataFrame(all_edges)

master_df = master_df.sort_values(by=['source', 'target']).reset_index(drop=True)

master_df.to_csv(f'./all_filtered_edges_across_tissues_{top}.csv', index=False)
print("merge complete!")
