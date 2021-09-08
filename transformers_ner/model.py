from transformers import BertConfig, BertModel
import torch.nn as nn
import torch.nn.functional as F

class BertNERModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        config: BertConfig = None,
        pretrained_path: str = None
    ):
        super().__init__()
        if not pretrained_path:
            self.bert = BertModel(config)
        else:
            self.bert = BertModel.from_pretrained(pretrained_path)
        self.head = nn.Linear(self.bert.embeddings.word_embeddings.embedding_dim, num_classes)
        self.loss_fn = F.cross_entropy
        
    def forward(self, x):
        labels = None
        if "labels" in x:
            labels = x["labels"]
            x.pop("labels")
        output = self.bert(**x)
        x = output[0]
        x = self.head(x)
        if labels is not None:
            x = x.transpose(1, 2)
            output["loss"] = self.loss_fn(x, labels)
        return output