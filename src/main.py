

@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    reader = ConllReader()
    train_dataset = BERTDataset(model_name, reader.read("data/conll2003/train.txt"))
    valid_dataset = BERTDataset(model_name, reader.read("data/conll2003/dev.txt"))


    from transformers import DataCollatorForTokenClassification

    data_collator = DataCollatorForTokenClassification(train_dataset.tokenizer, pad_to_multiple_of=None)

    from torch.utils.data.dataloader import DataLoader

    BATCH_SIZE = 8

    train_dataloader = DataLoader(
        train_dataset, shuffle=False, collate_fn=data_collator, batch_size=BATCH_SIZE
    )
    valid_dataloader = DataLoader(
        valid_dataset, shuffle=False, collate_fn=data_collator, batch_size=BATCH_SIZE
    )


    NUM_CLASSES = 5

    model = BertNERModel(num_classes=NUM_CLASSES, pretrained_path="bert-base-cased")


    from transformers import AdamW

    LR = 1e-5
    WEIGHT_DECAY = 0.0

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LR)



    from transformers import get_linear_schedule_with_warmup

    NUM_EPOCHS = 5
    NUM_WARMUP_STEPS = 0
    DEVICE = "cuda:2"

    max_train_steps = NUM_EPOCHS * len(train_dataloader)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=NUM_WARMUP_STEPS,
        num_training_steps=max_train_steps,
    )

    model.to(DEVICE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        with tqdm(total=len(train_dataloader)) as t:
            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
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

