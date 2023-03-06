from composer import Trainer
from composer import optim as composer_optimisers
from datasets import load_dataset
import torch
from transformers import AutoTokenizer

from utils import data_prep as dp
from models import negOPT_base

opt_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m", use_fast=False)

casino = load_dataset("casino")


def main():
    data = dp.prepare_casino_data(casino)

    casino_train_data, casino_test_data = dp.generate_dataset_splits(
        tokenizer=opt_tokenizer, processed_casino=data
    )

    print(f"Training on {len(casino_train_data)} examples.")

    device = "cuda" if torch.has_cuda else "cpu"

    model = negOPT_base.MosaicNegOPTBase("facebook/opt-350m")
    trainer = Trainer(
        model=model,
        train_dataloader=casino_train_data,
        eval_dataloader=casino_test_data,
        optimizers=composer_optimisers.DecoupledAdamW(params=model.parameters()),
        max_duration=4,
        device=device,
        train_subset_num_batches=1,  # !!!
        save_folder="./models/negOPT_checkpoints/",
        save_filename="ep{epoch}.pt",
        save_latest_filename="latest",
        save_overwrite=True,
    )

    trainer.fit()

    torch.save(model.state_dict(), "models/negOPT_base.pt")


if __name__ == "__main__":
    main()
