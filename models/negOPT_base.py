from composer.models import base as composer_models
from composer import metrics as composer_metrics
import torch.nn.functional as F
import transformers


class MosaicNegOPTBase(composer_models.ComposerModel):
    """Learning the basics of MosiacML by creating my own (model specific) huggingface model wrapper.

    Will use this as a framework for custom models completing the same task later."""

    def __init__(self, pretrained_model: str):
        super().__init__()
        self.model = transformers.OPTForCausalLM.from_pretrained(pretrained_model)
        self.perplex = composer_metrics.Perplexity()

    def loss(self, outputs, batch, *args, **kwargs):
        """Accepts the outputs from forward() and the batch"""
        # Flatten logits and labels (required by F.cross_entropy).
        _, _, labels = batch
        flat_logits = outputs.view(-1, outputs.size(-1))
        flat_targets = labels.view(-1)

        return F.cross_entropy(flat_logits, flat_targets)

    def get_metrics(self, is_train):
        return {"Perplexity": self.perplex}

    def forward(self, batch):

        input_ids, attention_mask, _ = batch

        y = self.model(input_ids, attention_mask).logits
        return F.log_softmax(y, dim=1)

    def eval_forward(self, batch, outputs=None):
        return outputs if outputs is not None else self.forward(batch)

    def update_metric(self, batch, outputs, metric) -> None:
        # Flatten logits and targets (required by F.cross_entropy).
        _, _, labels = batch
        flat_logits = outputs.view(-1, outputs.size(-1))
        flat_targets = labels.view(-1)
        metric.update(flat_logits, flat_targets)
