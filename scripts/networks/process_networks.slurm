#!/bin/bash
#SBATCH --job-name=process_networks          # create a short name for your job
#SBATCH --output=slurm-%j.out  # stdout file
#SBATCH --nodes=1                 # request one node
#SBATCH --time=1-12:00         # set the maximum runtime (1 day)
#SBATCH --mem=64G                 # request 16GB of memory (adjust as needed)

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

TOP_PCT=3.0
DAT_DIR="$ROOT_DIR/data/dat_networks"
NODES_FILE="$ROOT_DIR/resources/common_nodes.txt"

TMP_DIR="${DAT_DIR}/tmp"
mkdir -p "$TMP_DIR"

TOP_PCT_LABEL=$(printf "%.0f" "$TOP_PCT")

for file in "$DAT_DIR"/*.dat; do
  if [[ "$file" == *_filtered_top${TOP_PCT_LABEL}.dat ]]; then
    echo "Skipping $file (already filtered for TOP_PCT_LABEL=$TOP_PCT_LABEL)"
    continue
  fi

  if [[ "$file" == *filtered* ]]; then
    echo "Skipping $file (contains 'filtered' in name)"
    continue
  fi

  filename=$(basename "$file")
  tissue="${filename%.dat}"
  outfile="${DAT_DIR}/${tissue}_filtered_top${TOP_PCT_LABEL}.dat"

  if [ ! -f "$outfile" ]; then
    echo "Processing $tissue..."

    # Step 1: Filter only edges where both nodes are in the known node list
    awk 'NR==FNR {nodes[$1]; next} ($1 in nodes) && ($2 in nodes)' \
"$NODES_FILE" "$file" > "${TMP_DIR}/valid_edges.tmp"

    # Step 2: Count total valid edges
    total=$(wc -l < "${TMP_DIR}/valid_edges.tmp")
    top_n=$(awk -v t="$total" -v p="$TOP_PCT" 'BEGIN { printf "%d", t * (p / 100.0) }')
    [[ $top_n -lt 1 ]] && top_n=1  # Ensure at least one edge

    # Step 3: Sort valid edges by weight, then keep top 1% and ties
    sort -k3,3gr "${TMP_DIR}/valid_edges.tmp" | tee "${TMP_DIR}/sorted_edges.tmp" | 
    awk -v top_n="$top_n" '
      NR == top_n { cutoff = $3 }
      NR <= top_n || $3 == cutoff
    ' > "$outfile"

    echo "Filtered top ${TOP_PCT}% (including ties) from $file → $outfile"
    rm "${TMP_DIR}/valid_edges.tmp" "${TMP_DIR}/sorted_edges.tmp"

  else
    echo "Skipping $file (already filtered output exists)"
  fi
done