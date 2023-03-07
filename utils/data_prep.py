from collections import defaultdict
from matplotlib import pyplot as plt
import torch
from torch.utils import data as data_utils
import typing as tp
import transformers
import tqdm


class CasinoDatasetIntegrated:
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        self_prompts: tp.List[str],
        chat_logs: tp.List[str],
        batch_size: int = 1,
        max_len: int = 512,
    ):
        """
        Others
            - https://lightning-transformers.readthedocs.io/en/latest/tasks/nlp/language_modeling.html
        """
        combined_inputs = [
            self_prompt + chat_log
            for self_prompt, chat_log in zip(self_prompts, chat_logs)
        ]

        self.inputs, self.labels = [], []
        for i in range(0, len(self_prompts), batch_size):
            # Tokenize this input batch.
            batch_inputs = tokenizer(
                combined_inputs[i, i + batch_size],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_len,
            )

            self.inputs.append(batch_inputs)

            # We will not be predicting the first token, so remove it and add extra padding to the end.
            # ? Is this the correct way to go about this?
            batch_labels = torch.concat(
                (
                    batch_inputs["input_ids"][:, 1:],
                    torch.full(
                        size=(batch_inputs["input_ids"].size(0), 1),
                        fill_value=tokenizer.pad_token_id,
                    ),
                ),
                dim=1,
            )

            self.inputs.append(batch_inputs)
            self.labels.append(batch_labels)

    def __getitem__(self, idx):
        input_ids = self.inputs["input_ids"][idx]
        attention_mask = self.inputs["attention_mask"][idx]

        # ! Unsqueezing was necessary when inputs were not batched, should be okay to remove now.
        return {
            "input_ids": input_ids,  # .unsqueeze(0),
            "attention_mask": attention_mask,  # .unsqueeze(0),
            "labels": self.labels[idx],  # .unsqueeze(0),
        }

    def __len__(self):
        return len(self.inputs["input_ids"])


class IterableCasinoDataset(data_utils.IterableDataset):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        self_prompts: tp.List[str],
        chat_logs: tp.List[str],
        max_len: int = 512,
    ):
        """
        Others
            - https://lightning-transformers.readthedocs.io/en/latest/tasks/nlp/language_modeling.html
        """
        combined_inputs = [
            self_prompt + chat_log
            for self_prompt, chat_log in zip(self_prompts, chat_logs)
        ]

        self.inputs = tokenizer(
            combined_inputs,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_len,
        )

        # We will not be predicting the first token, so remove it and add extra padding to the end.
        # ? Is this the correct way to go about this?
        self.labels = torch.concat(
            (
                self.inputs["input_ids"][:, 1:],
                torch.full(
                    size=(self.inputs["input_ids"].size(0), 1),
                    fill_value=tokenizer.pad_token_id,
                ),
            ),
            dim=1,
        )

    def __iter__(self):
        return iter(
            zip(self.inputs["input_ids"], self.inputs["attention_mask"], self.labels)
        )

    def __len__(self):
        return len(self.inputs["input_ids"])


def prepare_casino_row(
    row: dict,
    sep_token: str,
    bot_name: str = "Bot",
    agent_name: str = "Agent",
) -> tp.List:
    #! CAUSAL MODELLING THROUGH DEAL OUTCOME AS INPUT
    data = defaultdict(list)
    for bot_id, info in row["participant_info"].items():
        bot_desires = """Bot Desires:
Low: {low_value_issue} - "{low_value_reason}"
Medium: {medium_value_issue} - "{medium_value_reason}"
High: {high_value_issue} - "{high_value_reason}"
""".format(
            low_value_issue=info["value2issue"]["Low"],
            medium_value_issue=info["value2issue"]["Medium"],
            high_value_issue=info["value2issue"]["High"],
            low_value_reason=info["value2reason"]["Low"],
            medium_value_reason=info["value2reason"]["Medium"],
            high_value_reason=info["value2reason"]["High"],
        )

        bot_personality = """Bot Personality:
SVO: {svo}
extraversion: {extrav}
agreeableness: {agreea}
conscientiousness: {consci}
emotional-stability: {emotstab}
openness-to-experience: {ote}
""".format(
            svo=info["personality"]["svo"],
            extrav=info["personality"]["big-five"]["extraversion"],
            agreea=info["personality"]["big-five"]["agreeableness"],
            consci=info["personality"]["big-five"]["conscientiousness"],
            emotstab=info["personality"]["big-five"]["emotional-stability"],
            ote=info["personality"]["big-five"]["openness-to-experiences"],
        )

        bot_outcome = """Bot Outcome:
Points Scored: {points_scored}
Satisfaction: {satisfaction}
        """.format(
            points_scored=info["outcomes"]["points_scored"],
            satisfaction=info["outcomes"]["satisfaction"],
        )

        id_to_name = lambda id_: bot_name if id_ == bot_id else agent_name
        chat_logs = sep_token.join(
            [
                id_to_name(utterance["id"]) + ": " + utterance["text"]
                for utterance in row["chat_logs"]
            ]
        )

        # Include only desires and outcome state for self prompt.
        data["self_prompts"].append("\n\n".join([bot_desires, bot_outcome]))

        data["chat_logs"].append(chat_logs)
    return data


def prepare_casino_data(casino_raw: any) -> dict:
    data = defaultdict(list)
    for row in tqdm.tqdm(casino_raw["train"]):
        row_data = prepare_casino_row(row, sep_token="")
        data["self_prompts"].extend(row_data["self_prompts"])
        data["chat_logs"].extend(row_data["self_prompts"])
    return data


def generate_dataset_splits(
    tokenizer,
    processed_casino: dict,
    test_frac: float = 0.1,
    max_len: float = 512,
    batch_size: int = 1,
) -> tp.Tuple[CasinoDatasetIntegrated]:

    train_n = int((1 - test_frac) * len(processed_casino["self_prompts"]))

    casino_train_data = IterableCasinoDataset(
        tokenizer=tokenizer,
        self_prompts=processed_casino["self_prompts"][:train_n],
        chat_logs=processed_casino["chat_logs"][:train_n],
        max_len=max_len,
    )
    casino_test_data = IterableCasinoDataset(
        tokenizer=tokenizer,
        self_prompts=processed_casino["self_prompts"][train_n:],
        chat_logs=processed_casino["chat_logs"][train_n:],
        max_len=max_len,
    )

    return casino_train_data, casino_test_data


def plot_input_lengths(data, tokenizer):
    lengths = [
        len(tokenizer(self_prompt + chat_log)["input_ids"])
        for self_prompt, chat_log in zip(data["self_prompts"], data["chat_logs"])
    ]

    fig, ax = plt.subplots()
    ax.hist(lengths)
    plt.show()
