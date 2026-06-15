#!/bin/bash
# Create datasets for ACF CO24

create_acf_dataset() {
    local db_path="$1"
    local dataset="$2"
    echo "Creating ACF dataset for $dataset from $db_path"
    for type in tossup bonus; do
        python create_dataset.py \
            --db-path "$db_path" --type "$type" \
            --prefix "$dataset" --repo-id "$dataset" --config-name "$type" \
            --org qanta-challenge
    done
}

for entry in \
    "data/dbs/co24-cleaned.db acf-co24" \
    "data/dbs/nats25.db acf-nats25" \
    "data/dbs/regs25.db acf-regs25"; do
    set -- $entry
    echo "Processing entry: $1 $2..."
    # create_acf_dataset "$1" "$2"
done
