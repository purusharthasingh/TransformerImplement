import torch
import torch.nn as nn
import torch.nn.functional as F
from model import POSTaggingModel
import math
import sys

import random
torch.manual_seed(0)
random.seed(0)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, n_head=4, d_qkv=32, dropout=0.1, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_qkv = d_qkv

        # We provide these model parameters to give an example of a weight
        # initialization approach that we know works well for our tasks. Feel free
        # to delete these lines and write your own, if you would like.
        self.w_q = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
        self.w_k = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
        self.w_v = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
        self.w_o = nn.Parameter(torch.Tensor(n_head, d_qkv, d_model))
        nn.init.xavier_normal_(self.w_q)
        nn.init.xavier_normal_(self.w_k)
        nn.init.xavier_normal_(self.w_v)
        nn.init.xavier_normal_(self.w_o)

        # The hyperparameters given as arguments to this function and others
        # should be sufficient to reach the score targets for this project

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        """Runs the multi-head self-attention layer.

        Args:
          x: the input to the layer, a tensor of shape [batch size, length, d_model]
          mask: a mask for disallowing attention to padding tokens. You will need to
                construct the mask yourself further on in this notebook. You may
                implement masking in any way; there is no requirement that you use
                a particular form for the mask object.
        Returns:
          A single tensor containing the output from this layer
        """
        """YOUR CODE HERE"""
        # Implementation tip: using torch.einsum will greatly simplify the code that
        # you need to write.

        # Apply linear projections to convert the feature vector at each token into separate vectors for the query, key, and value.
        q = torch.einsum('abc,ecf->aebf', x, self.w_q)
        k = torch.einsum('abc,ecf->aebf', x, self.w_k)
        v = torch.einsum('abc,ecf->aebf', x, self.w_v)
        
        # Apply attention, scaling the logits by 1 / d_{kqv} .
        logits = torch.einsum('abcd,abed->abce', q, k)
        scaled_logits = logits / math.sqrt(self.d_qkv)

        # Ensure proper masking, such that padding tokens are never attended to.
        # Create a tensor based on the mask
        if torch.cuda.is_available():
            mask = mask.cuda()
            mask = torch.where(mask==True, torch.zeros(mask.shape).cuda(), -1e9 * torch.ones(mask.shape).cuda())
        else:
            mask = torch.where(mask==True, torch.zeros(mask.shape), -1e9 * torch.ones(mask.shape))
        # mask = torch.where(mask==True, torch.zeros(mask.shape), -1e9 * torch.ones(mask.shape))
        # Reshape the tensor to match the shape of the logits
        mask = mask.unsqueeze(1).repeat(1, mask.shape[1], 1)
        mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        scaled_logits = scaled_logits + mask
        
        #Attention with sotmax and dropout
        if torch.cuda.is_available():
            attention = self.dropout(torch.softmax(scaled_logits, dim=-1)).cuda()
        else:
            attention = self.dropout(torch.softmax(scaled_logits, dim=-1)) 

        # The result sums over all the attention heads 
        prob = torch.einsum('abcd,abde->abce', attention, v) 
        res = torch.sum(torch.einsum('abcd,bde->abce', prob, self.w_o), 1)
        
        # Normalization and dropout 
        res = self.norm(self.dropout(res)+x)
        return res
        
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """YOUR CODE HERE"""
        # h1 = relu (w1(x))
        h_1 = self.relu(self.w_1(x))
        # h2 = w2(h1)
        h_2 = self.w_2(h_1)

        # Dropout and normalization 
        res = self.dropout(h_2)
        res = self.layer_norm(res + x)
        return res



class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, d_ff=1024, n_layers=4, n_head=4, d_qkv=32,
                 dropout=0.1):
        super().__init__()
        # Implementation tip: if you are storing nn.Module objects in a list, use
        # nn.ModuleList. If you use assignment statements of the form
        # `self.sublayers = [x, y, z]` with a plain python list instead of a
        # ModuleList, you might find that none of the sub-layer parameters are
        # trained.

        """YOUR CODE HERE"""
        self.n_layers = n_layers
        self.network = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.attention = MultiHeadAttention(d_model, n_head, d_qkv, dropout)


    def forward(self, x, mask):
        """Runs the Transformer encoder.

        Args:
          x: the input to the Transformer, a tensor of shape
             [batch size, length, d_model]
          mask: a mask for disallowing attention to padding tokens. You will need to
                construct the mask yourself further on in this notebook. You may
                implement masking in any way; there is no requirement that you use
                a particular form for the mask object.
        Returns:
          A single tensor containing the output from the Transformer
        """

        """YOUR CODE HERE"""
        # Iteratively goes through the networks n layers building on the previous layer
        for i in range(self.n_layers):
            x = self.attention(x, mask)
            x = self.network(x)
        return x

class AddPositionalEncoding(nn.Module):
    def __init__(self, d_model=256, input_dropout=0.1, timing_dropout=0.1,
                 max_len=512):
        super().__init__()
        self.timing_table = nn.Parameter(torch.FloatTensor(max_len, d_model))
        nn.init.normal_(self.timing_table)
        self.input_dropout = nn.Dropout(input_dropout)
        self.timing_dropout = nn.Dropout(timing_dropout)

    def forward(self, x):
        """
        Args:
          x: A tensor of shape [batch size, length, d_model]
        """
        x = self.input_dropout(x)
        timing = self.timing_table[None, :x.shape[1], :]
        timing = self.timing_dropout(timing)
        return x + timing


class TransformerPOSTaggingModel(POSTaggingModel):
    def __init__(self, vocab, PARTS_OF_SPEECH):
        super().__init__()
        d_model = 256
        self.add_timing = AddPositionalEncoding(d_model)
        self.encoder = TransformerEncoder(d_model)
        
        """more starting code."""
        self.PAD_ID = vocab.PieceToId("<pad>")
        """YOUR CODE HERE."""

        self.embeddings = nn.Embedding(vocab.GetPieceSize(), d_model)

    def encode(self, batch):
        """
        Args:
          batch: an input batch as a dictionary; the key 'ids' holds the vocab ids
            of the subword tokens in a tensor of size [batch_size, sequence_length]
        Returns:
          A single tensor containing logits for each subword token
            You don't need to filter the unlabeled subwords - this is handled by our
            code above.
        """

        # Implementation tip: you will want to use another normalization layer
        # between the output of the encoder and the final projection layer

        """YOUR CODE HERE."""
        ids = batch['ids']
        mask = ids != self.PAD_ID
        
        # Takes the embeddings and adds the positional encoding along before passing to the transformer
        embeddings = self.embeddings(ids)
        out = self.add_timing(embeddings)
        out = self.encoder(out, mask)
        return out
