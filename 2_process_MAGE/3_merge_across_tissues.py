import os
import pandas as pd

data_dir = '/mnt/home/aaggarwal/ceph/gates_proj/MAGE_networks/xgboost_mage_2025_networks'
top = 'top3'
print(f'processing {top} files:')

edge_dict = {}

filtered_files = [f for f in os.listdir(data_dir) if f.endswith(f'_filtered_{top}.dat')]

for f in filtered_files:
    tissue_name = f.replace(f'_filtered_{top}.dat', '')
    print(f"processing {f} as tissue: {tissue_name}", flush=True)

    df = pd.read_csv(os.path.join(data_dir, f), sep='\t', header=None, names=['source', 'target', 'weight'])

    df = df[df['source'] != df['target']]

    src = df['source']
    tgt = df['target']
    df['source'] = src.combine(tgt, min)
    df['target'] = src.combine(tgt, max)

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

output_path = os.path.join(data_dir, f'all_filtered_edges_across_tissues_{top}.csv')
master_df.to_csv(output_path, index=False)

print("merge complete!", flush=True)
