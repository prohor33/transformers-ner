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


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    logger = logging.getLogger("tramsformers-ner")

    logger.info("Working directory : {}".format(os.getcwd()))
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    reader = ConllReader()
    train_dataset = BERTDataset(cfg.model_name, reader.read(to_absolute_path(cfg.train_path)))
    valid_dataset = BERTDataset(cfg.model_name, reader.read(to_absolute_path(cfg.valid_path)))

    data_collator = DataCollatorForTokenClassification(train_dataset.tokenizer, pad_to_multiple_of=None)

    train_dataloader = DataLoader(
        train_dataset, shuffle=False, collate_fn=data_collator, batch_size=cfg.batch_size
    )
    valid_dataloader = DataLoader(
        valid_dataset, shuffle=False, collate_fn=data_collator, batch_size=cfg.batch_size
    )

    model = BertNERModel(num_classes=cfg.num_classes, pretrained_path=cfg.model_name)

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

    for epoch in range(cfg.num_epochs):
        model.train()
        with tqdm(total=len(train_dataloader)) as t:
            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to(cfg.device) for k, v in batch.items()}
                outputs = model(batch)
                loss = outputs.loss
                t.set_description("{:10.6f}".format(loss.item()))
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                t.update()

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]


if __name__ == "__main__":
    main()
