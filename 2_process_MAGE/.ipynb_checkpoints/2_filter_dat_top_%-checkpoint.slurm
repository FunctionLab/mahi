#!/bin/bash
#SBATCH --job-name=1%          # create a short name for your job
#SBATCH --output=slurm-%A.%a.out  # stdout file
#SBATCH --nodes=1                 # request one node
#SBATCH --time=1-00:00:00         # set the maximum runtime (1 day)
#SBATCH --mem=64G                 # request 16GB of memory (adjust as needed)
#SBATCH --mail-type=begin         # send mail when job begins
#SBATCH --mail-type=end           # send mail when job ends
#SBATCH --mail-type=fail          # send mail if job fails
#SBATCH --mail-user=aa8417@princeton.edu

for file in *.dat; do
  if [[ "$file" == *_filtered_top1.dat ]]; then
    echo "Skipping $file (already filtered)"
    continue
  fi

  outfile="${file%.dat}_filtered_top1.dat"

  if [ ! -f "$outfile" ]; then
    echo "Processing $file..."

    # Step 1: Filter only edges where both nodes are in the known node list
    awk 'NR==FNR {nodes[$1]; next} ($1 in nodes) && ($2 in nodes)' 0_all_nodes_18579.txt "$file" > /tmp/valid_edges_1.tmp

    # Step 2: Count total valid edges
    total=$(wc -l < /tmp/valid_edges_1.tmp)
    top_n=$(awk -v t="$total" 'BEGIN { printf "%d", t * 0.01 }')
    [[ $top_n -lt 1 ]] && top_n=1  # Ensure at least one edge

    # Step 3: Sort valid edges by weight, then keep top 1% and ties
    sort -k3,3gr /tmp/valid_edges_1.tmp | tee /tmp/sorted_edges_1.tmp | 
    awk -v top_n="$top_n" '
      NR == top_n { cutoff = $3 }
      NR <= top_n || $3 == cutoff
    ' > "$outfile"

    echo "Filtered top 1% (including ties) from $file → $outfile"
    rm /tmp/valid_edges_1.tmp /tmp/sorted_edges_1.tmp

  else
    echo "Skipping $file (already filtered output exists)"
  fi
done

