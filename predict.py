
import json
import hydra
import logging
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import DataCollatorForTokenClassification
from transformers.models.bert.configuration_bert import BertConfig
from omegaconf import OmegaConf
import os
from os import path
from hydra.utils import get_original_cwd, to_absolute_path
from transformers_ner.utils import bpe_to_tokens
from transformers_ner.model import BertNERModel
from transformers_ner.dataset import BERTDataset


@hydra.main(config_path="conf", config_name="predict_config")
def main(cfg):
    logger = logging.getLogger("tramsformers-ner")

    logger.info("Working directory : {}".format(os.getcwd()))
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    sample = "Hi from Russia!"
    tokens = sample.split()
    sequence = (tokens, ["O"]*len(tokens))
    sequences = [sequence]

    with open(to_absolute_path(path.join(cfg.load_path, cfg.label_to_index)), "r") as f:
        label_to_index = json.load(f)

    dataset = BERTDataset(
        cfg.model_name,
        sequences,
        label_to_index=label_to_index
    )

    data_collator = DataCollatorForTokenClassification(dataset.tokenizer)

    dataloader = DataLoader(
        dataset, shuffle=False, collate_fn=data_collator, batch_size=cfg.batch_size
    )

    bert_config = BertConfig.from_json_file(to_absolute_path(path.join(cfg.load_path, cfg.config)))
    model = BertNERModel(num_classes=len(label_to_index), config=bert_config)

    model.load_state_dict(torch.load(to_absolute_path(path.join(cfg.load_path, cfg.model))))
    model.to(cfg.device)

    model.eval()
    labels_pred = []
    labels_gold = []
    for step, batch in enumerate(dataloader):
        labels = batch["labels"]
        batch = {k: v.to(cfg.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs, logits = model(batch)
            predictions = logits.argmax(dim=1)
        predictions = predictions.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        labels_pred_batch, labels_gold_batch = bpe_to_tokens(predictions, labels, dataset.idx_to_label())
        labels_pred.extend(labels_pred_batch)
        labels_gold.extend(labels_gold_batch)
    
    logger.info(tokens)
    logger.info(labels_pred)


if __name__ == "__main__":
    main()