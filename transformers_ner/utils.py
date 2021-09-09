def bpe_to_tokens(y_pred, y_gold, idx_to_label):
    # Remove ignored index (special tokens)
    labels_pred = []
    labels_gold = []
    for pred, gold in zip(y_pred, y_gold):
        labels_pred.extend([idx_to_label[p] for (p, g) in zip(pred, gold) if g != -100])
        labels_gold.extend([idx_to_label[g] for (p, g) in zip(pred, gold) if g != -100])
    return labels_pred, labels_gold