from composer import Trainer
from composer import optim as composer_optimisers
from datasets import load_dataset
import torch
from torch.utils import data as data_utils
from transformers import AutoTokenizer

from utils import data_prep as dp
from models import negOPT_base

opt_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m", use_fast=False)

casino = load_dataset("casino")

BATCH_SIZE = 1


def main():
    data = dp.prepare_casino_data(casino)

    casino_train_data, casino_test_data = dp.generate_dataset_splits(
        tokenizer=opt_tokenizer, processed_casino=data
    )

    print(f"Training on {len(casino_train_data)} examples.")

    device = "gpu" if torch.has_cuda else "cpu"

    model = negOPT_base.MosaicNegOPTBase("facebook/opt-350m")

    train_dataloader = data_utils.DataLoader(
        casino_train_data,
        batch_size=BATCH_SIZE,
    )

    eval_dataloader = data_utils.DataLoader(casino_test_data, batch_size=BATCH_SIZE)

    optimizer = composer_optimisers.DecoupledAdamW(params=model.parameters())

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizers=optimizer,
        max_duration=1,
        device=device,
        log_to_console=True,
        console_log_interval=1,
        save_folder="./models/negOPT_checkpoints/",
        save_filename="ep{epoch}.pt",
        save_latest_filename="latest.pt",
        save_overwrite=True,
        train_subset_num_batches=-1,  # !!!
    )

    trainer.fit()

    torch.save(model.state_dict(), "models/negOPT_base.pt")


if __name__ == "__main__":
    main()
