from transformers.models.bert.configuration_bert import BertConfig
from transformers_ner.utils import bpe_to_tokens
from transformers_ner.reader import ConllReader
from transformers_ner.dataset import BERTDataset
from transformers_ner.model import BertNERModel
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
import torch
from transformers import DataCollatorForTokenClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data.dataloader import DataLoader
import logging
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from sklearn.metrics import classification_report
import json


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    logger = logging.getLogger("tramsformers-ner")

    logger.info("Working directory : {}".format(os.getcwd()))
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    reader = ConllReader()
    train_dataset = BERTDataset(
        cfg.model_name,
        reader.read(to_absolute_path(cfg.train_path),
        samples_number=None)
    )
    label_to_index = train_dataset.label_to_index
    valid_dataset = BERTDataset(
        cfg.model_name,
        reader.read(to_absolute_path(cfg.valid_path),
        samples_number=None),
        label_to_index=label_to_index
    )
    with open("label_to_index.json", "w") as f:
        json.dump(train_dataset.label_to_index, f)

    data_collator = DataCollatorForTokenClassification(train_dataset.tokenizer, pad_to_multiple_of=None)

    train_dataloader = DataLoader(
        train_dataset, shuffle=False, collate_fn=data_collator, batch_size=cfg.batch_size
    )
    valid_dataloader = DataLoader(
        valid_dataset, shuffle=False, collate_fn=data_collator, batch_size=cfg.batch_size
    )

    if cfg.train_from_scratch:
        bert_config = BertConfig.from_json_file(to_absolute_path(cfg.from_scratch_bert_config))
        model = BertNERModel(num_classes=len(label_to_index), config=bert_config)
    else:
        model = BertNERModel(num_classes=len(label_to_index), pretrained_path=cfg.model_name)
    model.bert.config.to_json_file("config.json")

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)

    max_train_steps = cfg.num_epochs * len(train_dataloader)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    model.to(cfg.device)

    best_f1_macro = 0.0

    for epoch in range(cfg.num_epochs):
        logger.info(f"Epoch {epoch}:")
        model.train()
        with tqdm(total=len(train_dataloader)) as t:
            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to(cfg.device) for k, v in batch.items()}
                outputs, _ = model(batch)
                loss = outputs.loss
                t.set_description("{:10.6f}".format(loss.item()))
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                t.update()

        model.eval()
        labels_pred = []
        labels_gold = []
        for step, batch in enumerate(valid_dataloader):
            labels = batch["labels"]
            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs, logits = model(batch)
                predictions = logits.argmax(dim=1)
            predictions = predictions.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            labels_pred_batch, labels_gold_batch = bpe_to_tokens(predictions, labels, train_dataset.idx_to_label())
            labels_pred.extend(labels_pred_batch)
            labels_gold.extend(labels_gold_batch)

        report_str = classification_report(labels_gold, labels_pred)
        logger.info(f"\n{report_str}")

        report = classification_report(labels_gold, labels_pred, output_dict=True)
        f1_macro = report["macro avg"]["f1-score"]
        if f1_macro > best_f1_macro:
            logger.info(f"Saving best model to: {cfg.save_model_path}")
            torch.save(model.state_dict(), cfg.save_model_path)
            with open("report.json", "w") as f:
                json.dump(report, f)


if __name__ == "__main__":
    main()

