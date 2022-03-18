import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Now it's time to build a model. At a high level, the model will encode the sentence using a Transformer architecture, then project to a softmax over the vocabulary at each word position. We've implemented the overall model framework already, including computing the softmax cross-entropy loss for training the tagger.
"""


class POSTaggingModel(nn.Module):
    def encode(self, batch):
        # you will override this function in a subclass below
        raise NotImplementedError()

    def compute_loss(self, batch):
        logits = self.encode(batch)
        logits = logits.reshape((-1, logits.shape[-1]))
        labels = batch['labels'].reshape((-1,))
        res = F.cross_entropy(
            logits, labels, ignore_index=-1, reduction='mean')
        return res

    def get_validation_metric(self, data_loader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in data_loader:
                mask = (batch['labels'] != -1)
                predicted_labels = self.encode(batch).argmax(-1)
                predicted_labels = predicted_labels[mask]
                gold_labels = batch['labels'][mask]
                correct += (predicted_labels == gold_labels).sum().item()
                total += gold_labels.shape[0]
        return correct / total


class BaselineModel(POSTaggingModel):
    def __init__(self, vocab, PARTS_OF_SPEECH):
        super().__init__()
        self.lookup = nn.Embedding(vocab.GetPieceSize(), len(PARTS_OF_SPEECH))

    def encode(self, batch):
        ids = batch['ids']
        return self.lookup(ids)
