# Yet another BERT NER implementation with 🤗 [Transformers](https://github.com/huggingface/transformers)

Results on Conll after 5 epochs on validation

entity | precision | recall | f1-score | support
------------ | ------------- | ------------- | ------------- | -------------
LOC         |   0.97   |   0.97   |   0.97   |   2094
MISC        |   0.93  |    0.91  |    0.92   |   1268
 ORG        |   0.95  |    0.95  |    0.95   |   2092
 PER        |   0.98  |    0.99  |    0.98   |   3149
   O        |   1.00  |    1.00  |    1.00   |  42759
accuracy    |   -     |     -    |    0.99   |  51362
macro avg   |      0.97   |   0.96   |   0.97   |  51362
weighted avg  |     0.99    |  0.99  |    0.99    | 51362

TODO:
- [ ] Add predict
- [ ] Add logging to tensorboard
- [ ] Try different datasets
- [ ] Make experiments with different hyperparams
- [ ] Try to tweak model architecture
