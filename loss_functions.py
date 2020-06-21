import torch
import torch.nn as nn

def triplet_loss(emb_anchor, emb_positive, emb_negative, margin = 1):
	
	max_fn = nn.ReLU()
	positive_pair_loss = torch.sum((emb_anchor - emb_positive)**2, dim = 1)
	negative_pair_loss = torch.sum((emb_anchor - emb_negative)**2, dim = 1)
	loss = max_fn(margin + positive_pair_loss - negative_pair_loss)
	loss = torch.mean(loss)
	return loss