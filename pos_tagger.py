###### Transformer for Part-of-Speech Tagging #####
# In this task, we will label each word token in a sentence or corpus with a part-of-speech tag. We provide you with all of the data processing code and a baseline model. You will implement the model using the Transformer architecture in transformer.py
###################################################
from transformer import TransformerPOSTaggingModel
from model import BaselineModel
from torch.nn.utils.rnn import pad_sequence
import torch
import sentencepiece
import argparse
import tqdm
import nltk
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
nltk.download('tagsets')


device = torch.device("cuda")
print("Using device:", device)


def load_data():
    """
    The files include one tree per line. 
    We'll use the BracketParseCorpusReader from nltk to load the data.
    """
    data_reader = BracketParseCorpusReader('./data', ['train', 'dev', 'test'])

    return data_reader


def build_vocab(data_reader):
    # Build Vocabulary
    """
    We first extract the sentences alone from the data and construct a subword vocabulary. We use a subword vocabulary because it allows us to largely avoid the issue of having unknown words at test time.
    """
    with open('sentences.txt', 'w') as f:
        for sent in data_reader.sents('train'):
            f.write(' '.join(sent) + '\n')

    args = {
        "pad_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "unk_id": 3,
        "input": "sentences.txt",
        "vocab_size": 16000,
        "model_prefix": "ptb",
    }
    combined_args = " ".join("--{}={}".format(key, value)
                             for key, value in args.items())
    sentencepiece.SentencePieceTrainer.Train(combined_args)

    vocab = sentencepiece.SentencePieceProcessor()
    vocab.Load("ptb.model")

    return vocab


def encode_sentence(vocab, sent):
    """Prepares a sentence for input to the model, including subword tokenization.

    Args:
        sent: a list of words (each word is a string)
    Returns:
        A tuple (ids, is_word_end).
        ids: a list of token ids in the subword vocabulary
        is_word_end: a list with elements of type bool, where True indicates that
                    the word piece at that position is the last within its word.
    """
    ids = []
    is_word_end = []
    for word in sent:
        word_ids = vocab.EncodeAsIds(word)
        ids.extend(word_ids)
        is_word_end.extend([False] * (len(word_ids) - 1) + [True])
    return ids, is_word_end


def get_pos_vocab(data_reader):
    all_pos = set()
    for sent in data_reader.tagged_sents('train'):
        for word, pos in sent:
            all_pos.add(pos)
    return sorted(all_pos)


class POSTaggingDataset(torch.utils.data.Dataset):
    """
    The POSTaggingDataset object defined below is a PyTorch Dataset object for this task.

    Each example in the dataset is a feature dictionary, consisting of word piece ids, and corresponding label ids (labels). We associate a word's label with the last subword. Any remaining subwords, as well as special tokens like the start token or padding token, will have a label of -1 assigned to them. This will signal that we shouldn't compute a loss for that label.

    We also define a collate function that takes care of padding when examples are batched together.
    """

    def __init__(self, split, data_reader, vocab, PARTS_OF_SPEECH):
        assert split in ('train', 'dev', 'test')
        self.sents = data_reader.tagged_sents(split)
        self.vocab = vocab
        self.PAD_ID = vocab.PieceToId("<pad>")
        self.BOS_ID = vocab.PieceToId("<s>")
        self.EOS_ID = vocab.PieceToId("</s>")
        self.UNK_ID = vocab.PieceToId("<unk>")
        self.PARTS_OF_SPEECH = PARTS_OF_SPEECH
        if split == 'train':
            # To speed up training, we only train on short sentences.
            self.sents = [sent for sent in self.sents if len(sent) <= 40]

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        sent = self.sents[index]
        ids, is_word_end = encode_sentence(
            self.vocab, [word for word, pos in sent])
        ids = [self.BOS_ID] + ids + [self.EOS_ID]
        is_word_end = [False] + is_word_end + [False]
        ids = torch.tensor(ids)
        is_word_end = torch.tensor(is_word_end)
        labels = torch.full_like(ids, -1)
        labels[is_word_end] = torch.tensor(
            [self.PARTS_OF_SPEECH.index(pos) for word, pos in sent])
        return {'ids': ids, 'labels': labels}

    @staticmethod
    def collate(batch, PAD_ID):
        ids = pad_sequence(
            [item['ids'] for item in batch],
            batch_first=True, padding_value=PAD_ID)
        labels = pad_sequence(
            [item['labels'] for item in batch],
            batch_first=True, padding_value=-1)
        return {'ids': ids.to(device), 'labels': labels.to(device)}


def train(model, data_reader, vocab, PARTS_OF_SPEECH,
          num_epochs, batch_size, model_file, learning_rate, dataset_cls=POSTaggingDataset):
    """Train the model and save its best checkpoint.

    Model performance across epochs is evaluated on the validation set. The best
    checkpoint obtained during training will be stored on disk and loaded back
    into the model at the end of training.
    """
    PAD_ID = vocab.PieceToId("<pad>")

    dataset = dataset_cls('train', data_reader, vocab, PARTS_OF_SPEECH)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: dataset.collate(batch, PAD_ID))

    dev_dataset = dataset_cls('dev', data_reader, vocab, PARTS_OF_SPEECH)
    dev_data_loader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=8, collate_fn=lambda batch: dataset.collate(batch, PAD_ID))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(data_loader),
        pct_start=0.02,  # Warm up for 2% of the total training time
    )

    best_metric = 0.0
    for epoch in tqdm.trange(num_epochs, desc="training", unit="epoch"):
        with tqdm.tqdm(
                data_loader,
                desc="epoch {}".format(epoch + 1),
                unit="batch",
                total=len(data_loader)) as batch_iterator:
            model.train()
            total_loss = 0.0
            for i, batch in enumerate(batch_iterator, start=1):
                optimizer.zero_grad()
                loss = model.compute_loss(batch)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step()
                batch_iterator.set_postfix(mean_loss=total_loss / i)
            validation_metric = model.get_validation_metric(dev_data_loader)
            batch_iterator.set_postfix(
                mean_loss=total_loss / i,
                validation_metric=validation_metric)
            if validation_metric > best_metric:
                print(
                    "Obtained a new best validation metric of {:.3f}, saving model "
                    "checkpoint to {}...".format(validation_metric, model_file))
                torch.save(model.state_dict(), model_file)
                best_metric = validation_metric
    print("Reloading best model checkpoint from {}...".format(model_file))
    model.load_state_dict(torch.load(model_file))
    print("Best validation accuracy", best_metric)


def predict_tags(data_reader, vocab, PARTS_OF_SPEECH, tagging_model, split, limit=None):
    assert split in ('dev', 'test')
    sents = data_reader.sents(split)
    dataset = POSTaggingDataset(split, data_reader, vocab, PARTS_OF_SPEECH)
    PAD_ID = vocab.PieceToId("<pad>")
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=False, collate_fn=lambda batch: dataset.collate(batch, PAD_ID))
    tagging_model.eval()
    pred_tagged_sents = []
    with torch.no_grad():
        for batch in data_loader:
            mask = (batch['labels'] != -1)
            predicted_labels = tagging_model.encode(batch).argmax(-1)
            for i in range(batch['ids'].shape[0]):
                example_predicted_tags = [
                    PARTS_OF_SPEECH[label] for label in predicted_labels[i][mask[i]]]
                sent = sents[len(pred_tagged_sents)]
                assert len(sent) == len(example_predicted_tags)
                pred_tagged_sents.append(
                    list(zip(sent, example_predicted_tags)))
                if limit is not None and len(pred_tagged_sents) >= limit:
                    return pred_tagged_sents
    return pred_tagged_sents


def run_baseline(data_reader, vocab, PARTS_OF_SPEECH):
    """
    We can now train a very simple baseline model that learns a single parameter per part of speech tag.

    A classifier where each word is assigned its most-frequent tag from the training data (and unknown words are treated as nouns) has 92.2% validation accuracy with our current splits. However, with our subword vocabulary (and taking the last subword as representative of each word), the accuracy is instead 87.6%. The BaselineModel should achieve roughly this accuracy.
    """
    baseline_model = BaselineModel(vocab, PARTS_OF_SPEECH).to(device)
    train(baseline_model, data_reader, vocab, PARTS_OF_SPEECH,
          num_epochs=5, batch_size=64, model_file="baseline_model.pt", learning_rate=0.1)

    """
    Having trained the model, we can examine its predictions on an example from the validation set.
    """
    predict_tags(data_reader, vocab, PARTS_OF_SPEECH,
                 baseline_model, 'dev', limit=1)

    return


def run_transformer_tagger(data_reader, vocab, PARTS_OF_SPEECH):
    tagging_model = TransformerPOSTaggingModel(
        vocab, PARTS_OF_SPEECH).to(device)
    train(tagging_model, data_reader, vocab, PARTS_OF_SPEECH,
          num_epochs=8, batch_size=16, model_file="tagging_model.pt", learning_rate=8e-4)

    predict_tags(data_reader, vocab, PARTS_OF_SPEECH,
                 tagging_model, 'dev', limit=1)

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline")
    args = parser.parse_args()
    print(f"RUN: {vars(args)}")

    data_reader = load_data()
    vocab = build_vocab(data_reader)

    print("Vocabulary size:", vocab.GetPieceSize())
    print()

    """
    Although we would like to use a subword vocabulary to better handle rare words, the part-of-speech and parsing tasks are defined in terms of words, not subwords. After encoding a sentence at the subword level with an encoder (LSTM or Transformer), we will then move to the word level by selecting a single representation per word. In the encode_sentence function below, we will create a boolean mask to select from the last subword of every word. Tagging decisions will be made based on the vector associated with this last subword. (You may modify this to, for example, use the first word piece instead, though that seems to perform slightly worse at least for the baseline model.)
    """
    print("Examples of encoding using Vocab:")
    for sent in data_reader.sents('train')[:2]:
        indices, is_word_end = encode_sentence(vocab, sent)
        pieces = [vocab.IdToPiece(index) for index in indices]
        print(sent)
        print(pieces)
        print(vocab.DecodePieces(pieces))
        print(indices)
        print(vocab.DecodeIds(indices))
        print()

    """
    Now we turn our attention to the desired output from the model, namely a sequence of part of speech tags.
    We construct a part of speech tag vocabulary by iterating over all tags in the training data. Note that opening parentheses ( are escaped as -LRB- in the data format (and similarly ) is escaped as -RRB-)
    """

    PARTS_OF_SPEECH = get_pos_vocab(data_reader)

    print('PARTS_OF_SPEECH')
    print(PARTS_OF_SPEECH)
    print()

    """
    Train Models and Predict Tags on validation set
    """

    if args.model == "baseline":
        run_baseline(data_reader, vocab, PARTS_OF_SPEECH)
    else:
        run_transformer_tagger(data_reader, vocab, PARTS_OF_SPEECH)

    return


if __name__ == '__main__':
    main()
