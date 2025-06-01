import argparse
import os

from datasets import Dataset
from huggingface_hub import whoami

from utils.acf_datasets import create_bonus_dataset, create_tossup_dataset

# %%


def parse_arguments():
    parser = argparse.ArgumentParser(
        "Create Tossup Dataset and push on Huggingface Hub"
    )

    parser.add_argument(
        "--db-path", "-d", required=True, help="Path to the quizbowl database"
    )
    parser.add_argument(
        "--type",
        "-t",
        choices=["tossup", "bonus"],
        default="tossup",
        help="Type of the dataset (Quizbowl questions) to create.",
    )
    parser.add_argument(
        "--prefix",
        "-p",
        required=True,
        help="Prefix for the qid created for each question.",
    )
    parser.add_argument(
        "--repo-id",
        "-r",
        default=None,
        help="Repository ID on HuggingFace Hub without the username/org prefix.",
    )
    parser.add_argument(
        "--config-name",
        "-c",
        default="default",
        help="Optional config name for the dataset. If not provided, it will be the default config name.",
    )
    parser.add_argument(
        "--org",
        "-o",
        required=False,
        help="Optional org name to push it under. If not provided, it will be pushed under the user.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="If set, the dataset will be pushed as public. Otherwise, it will be private.",
    )
    return parser.parse_args()


def get_username_from_hf():
    info = whoami()
    return info["name"], [org["name"] for org in info.get("orgs", [])]


def resolve_repo_id(args):
    user, orgs = get_username_from_hf()
    if "/" in args.repo_id:
        raise SystemError(
            "Please provide a repository ID without the username/org prefix. "
            "The script will automatically prepend it based on your user/org."
        )

    if args.org and args.org not in orgs:
        raise SystemError(
            f"Provided org '{args.org}' does not match any orgs you belong to."
            f" Please check your org name. Available orgs: {orgs}"
        )
    if args.org:
        return f"{args.org}/{args.repo_id}"

    return f"{user}/{args.repo_id}"


if __name__ == "__main__":
    args = parse_arguments()
    if not args.repo_id:
        print(
            f"Repository ID not provided. It will be automatically generated based on the prefix '{args.prefix}' and type '{args.type}'."
        )
        args.repo_id = f"{args.prefix}-{args.type}"

    if not os.path.exists(args.db_path):
        raise FileNotFoundError(f"Database file {args.db_path} does not exist.")

    repo_id = resolve_repo_id(args)
    if args.type == "bonus":
        dataset = create_bonus_dataset(args.db_path, args.prefix)
    else:
        dataset = create_tossup_dataset(args.db_path, args.prefix)
    dataset.push_to_hub(
        repo_id, args.config_name, split="eval", private=not args.public
    )

# %%

# ds.push_to_hub("umdclip/acf-co24-bonuses", split="eval", private=True)
# %%

"""
Example usage:
python create_dataset.py \
    --db-path data/dbs/nats25.db --type tossup \
    --prefix acf-nats25 \
    --repo-id acf-nats25-tossups \
    --org qanta-challenge


python create_dataset.py \
    --db-path data/dbs/nats25.db --type tossup \
    --prefix acf-nats25 \
    --repo-id acf-nats25-tossups \
    --org qanta-challenge

python create_dataset.py \
    --db-path data/dbs/nats25.db --type tossup \
    --prefix acf-nats25 \
    --repo-id acf-nats25-tossups \
    --org qanta-challenge
"""
