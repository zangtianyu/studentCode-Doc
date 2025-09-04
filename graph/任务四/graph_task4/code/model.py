import logging
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from datetime import datetime

from dataloader import TestDataset

class TransE(nn.Module):
    def __init__(self, model_name, nentity, nrelation, args):
        super(TransE, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.lmbda = args.lmbda
        self.gamma = nn.Parameter(
            torch.Tensor([args.gamma]), 
            requires_grad=False
        )

        self.entity_dim = self.hidden_dim
        self.relation_dim = self.hidden_dim

        self.entity_embedding = nn.Embedding(nentity, self.entity_dim)
        self.relation_embedding = nn.Embedding(nentity, self.relation_dim)

        self.init_parameters()

    def init_parameters(self):
        if not self.args.use_init:
            nn.init.xavier_uniform_(self.entity_embedding.weight.data)
            nn.init.xavier_uniform_(self.relation_embedding.weight.data)

    def load_embedding(self, init_ent_embs, init_rel_embs):

        init_ent_embs = torch.from_numpy(init_ent_embs)
        init_rel_embs = torch.from_numpy(init_rel_embs)

        if self.args.cuda:
            init_ent_embs = init_ent_embs.cuda()
            init_rel_embs = init_rel_embs.cuda()

        self.entity_embedding.weight.data = init_ent_embs
        self.relation_embedding.weight.data = init_rel_embs
        
        print("Load form Embedding success !")

    def loss(self, positive_score, negative_score):
        if self.args.negative_adversarial_sampling:
            negative_score = (F.softmax(negative_score * self.args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score = F.logsigmoid(positive_score)

        positive_sample_loss = - positive_score.mean()
        negative_sample_loss = - negative_score.mean()
        loss = (positive_sample_loss + negative_sample_loss)/2
        if self.args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = self.args.regularization * (
                self.entity_embedding.norm(p = 3)**3 + 
                self.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
        return loss

    def forward(self, px, nx, py, ny):

        ph = self.entity_embedding(px[:,0])
        pr = self.relation_embedding(px[:,1])
        pt = self.entity_embedding(px[:,2])
        positive_score = self._calc(ph, pr, pt)

        nh = self.entity_embedding(nx[:,0])
        nr = self.relation_embedding(nx[:,1])
        nt = self.entity_embedding(nx[:,2])
        negative_score = self._calc(nh, nr, nt).reshape([-1, self.args.neg_ratio])

        return self.loss(positive_score, negative_score)

    def _calc(self, h, r, t):
        score = h + r - t
        score = self.gamma.item() - torch.norm(score, p=1, dim=1)
        return score.squeeze()

    def predict(self, x):

        h = self.entity_embedding(x[:, 0])
        r = self.relation_embedding(x[:, 1])
        t = self.entity_embedding(x[:, 2])

        score = self._calc(h, r, t)

        return score


  
class RotatE(nn.Module):  
    def __init__(self, model_name, nentity, nrelation, args):  
        super(RotatE, self).__init__()  
        self.model_name = model_name  
        self.nentity = nentity  
        self.nrelation = nrelation  
        self.args = args  
        self.embedding_dim = args.hidden_dim  
        self.gamma = nn.Parameter(  
            torch.Tensor([args.gamma]),   
            requires_grad=False  
        )  
  
        # Complex embeddings for entities and relations  
        self.entity_embedding = nn.Embedding(nentity, self.embedding_dim * 2)  
        self.relation_embedding = nn.Embedding(nrelation, self.embedding_dim * 2)  
  
        self.init_parameters()  
  
    def init_parameters(self):  
        if not self.args.use_init:  
            nn.init.xavier_uniform_(self.entity_embedding.weight.data)  
            nn.init.xavier_uniform_(self.relation_embedding.weight.data)  
  
    def forward(self, px, nx, py, ny):  
        # Split positive and negative triplets  
        ph, pr, pt = self.split_and_embed(px)  
        nh, nr, nt = self.split_and_embed(nx)  
  
        # Calculate scores  
        positive_score = self._calc(ph, pr, pt)  
        negative_scores = self._calc(nh, nr, nt).reshape([-1, self.args.neg_ratio])  
  
        return self.loss(positive_score, negative_scores)  
  
    def split_and_embed(self, triplets):  
        # Extract heads, relations, and tails from triplets  
        heads = triplets[:, 0]  
        relations = triplets[:, 1]  
        tails = triplets[:, 2]  
  
        # Embed entities and relations  
        embeddings = self.entity_embedding(torch.cat([heads, tails], dim=0))  
        relations_emb = self.relation_embedding(relations)  
  
        # Split embeddings into real and imaginary parts  
        heads_re, heads_im = embeddings[:heads.size(0), :self.embedding_dim], embeddings[:heads.size(0), self.embedding_dim:]  
        tails_re, tails_im = embeddings[heads.size(0):, :self.embedding_dim], embeddings[heads.size(0):, self.embedding_dim:]  
  
        return heads_re, heads_im, relations_emb, tails_re, tails_im  
  
    def _calc(self, h_re, h_im, r_emb, t_re, t_im):  
        # Split relation embeddings into real and imaginary parts  
        r_re, r_im = r_emb[:, :self.embedding_dim], r_emb[:, self.embedding_dim:]  
  
        # Perform rotation  
        rotated_h_re = h_re * torch.cos(r_im) - h_im * torch.sin(r_im)  
        rotated_h_im = h_re * torch.sin(r_im) + h_im * torch.cos(r_im)  
  
        # Calculate scores  
        score_re = rotated_h_re - t_re  
        score_im = rotated_h_im - t_im  
  
        # Euclidean distance  
        scores = torch.norm(torch.stack([score_re, score_im], dim=-1), dim=-1, p=2)  
  
        # Margin-based scoring  
        return self.gamma.item() - scores  
  
    def loss(self, positive_score, negative_scores):  
        # Margin-based ranking loss  
        positive_loss = F.relu(self.gamma.item() - positive_score).mean()  
        negative_loss = F.relu(negative_scores - self.gamma.item()).mean()  
  
        # Total loss  
        loss = positive_loss + negative_loss  
  
        if self.args.regularization != 0.0:  
            # Optionally add regularization  
            regularization = self.args.regularization * (  
                self.entity_embedding.weight.norm(p=2)**2 +  
                self.relation_embedding.weight.norm(p=2)**2  
            )  
            loss += regularization  
  
        return loss  
  
    # Optional: add a predict method if needed  
    def predict(self, x):  
        h_re, h_im, r_emb, t_re, t_im = self.split_and_embed(x)  
        scores = self._calc(h_re, h_im, r_emb, t_re, t_im)  
        return scores

  
class ConvE(nn.Module):
    def __init__(self, model_name, nentity, nrelation, args):  
        super(ConvE, self).__init__()  
        self.model_name = model_name  
        self.nentity = nentity  
        self.nrelation = nrelation  
        self.args = args  
        self.hidden_dim = args.hidden_dim  
        self.embedding_dim = args.embedding_dim    
        self.kernel_size = args.kernel_size  
        self.out_channels = args.out_channels  
        self.feature_width = args.feature_width
        self.gamma = nn.Parameter(  
            torch.Tensor([args.gamma]),   
            requires_grad=False  
        )  
        self.height = 2 * self.embedding_dim // self.feature_width
        self.width = self.feature_width
        if self.height * self.width != 2 * self.embedding_dim:
            raise ValueError("2*embedding_dim must be divisible by feature_width")
        self.conv = nn.Conv2d(1, self.out_channels, (self.kernel_size, self.kernel_size))
        flat_h = self.height - self.kernel_size + 1
        flat_w = self.width - self.kernel_size + 1
        if flat_h <= 0 or flat_w <= 0:
            raise ValueError("Invalid dimensions for convolution")
        flat_size = self.out_channels * flat_h * flat_w
        self.fc = nn.Linear(flat_size, self.hidden_dim)  
        self.dropout = nn.Dropout(0.3)
        if self.embedding_dim != self.hidden_dim:
            self.tail_proj = nn.Linear(self.embedding_dim, self.hidden_dim)
        else:
            self.tail_proj = nn.Identity()
        self.entity_embedding = nn.Embedding(nentity, self.embedding_dim)  
        self.relation_embedding = nn.Embedding(nrelation, self.embedding_dim)  
        self.init_parameters()  

    def init_parameters(self):  
        nn.init.xavier_uniform_(self.entity_embedding.weight.data)  
        nn.init.xavier_uniform_(self.relation_embedding.weight.data)  
        nn.init.xavier_uniform_(self.conv.weight)  
        nn.init.zeros_(self.conv.bias)  
        nn.init.xavier_uniform_(self.fc.weight)  
        nn.init.zeros_(self.fc.bias)  
        if self.embedding_dim != self.hidden_dim:
            nn.init.xavier_uniform_(self.tail_proj.weight)
            nn.init.zeros_(self.tail_proj.bias)

    def forward(self, px, nx, py, ny):  
        positive_score = self.predict(px)  
        negative_score = self.predict(nx).reshape(px.shape[0], self.args.neg_ratio)  
        return self.loss(positive_score, negative_score)  

    def loss(self, positive_score, negative_score):  
        if self.args.negative_adversarial_sampling:  
            negative_score = (F.softmax(negative_score * self.args.adversarial_temperature, dim = 1).detach()   
                              * F.logsigmoid(-negative_score)).sum(dim = 1)  
        else:  
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)  
        positive_score = F.logsigmoid(positive_score)  
        positive_sample_loss = - positive_score.mean()  
        negative_sample_loss = - negative_score.mean()  
        loss = (positive_sample_loss + negative_sample_loss)/2  
        if self.args.regularization != 0.0:  
            regularization = self.args.regularization * (  
                self.entity_embedding.norm(p = 3)**3 +   
                self.relation_embedding.norm(p = 3)**3  
            )  
            loss = loss + regularization  
        return loss  

    def predict(self, x):  
        heads = self.entity_embedding(x[:, 0])  
        rels = self.relation_embedding(x[:, 1])  
        tails = self.entity_embedding(x[:, 2])  
        concat = torch.cat([heads, rels], dim=1)  
        N = concat.shape[0]  
        reshaped = concat.view(N, 1, self.height, self.width)  
        conved = self.conv(reshaped)  
        conved = F.relu(conved)  
        conved = self.dropout(conved)  
        flat = conved.view(N, -1)  
        hidden = self.fc(flat)  
        hidden = F.relu(hidden)  
        hidden = self.dropout(hidden)  
        inner = (hidden * self.tail_proj(tails)).sum(dim=1)  
        score = self.gamma.item() + inner  
        return score  



def train_step(model, optimizer, train_iterator, args):
    '''
    A single train step. Apply back-propation and return the loss
    '''

    model.train()
    optimizer.zero_grad()

    positive_batch, negative_batch, yp_batch, yn_batch = next(train_iterator)

    if args.cuda:
        positive_batch = positive_batch.cuda()
        negative_batch = negative_batch.cuda()
        yp_batch = yp_batch.cuda()
        yn_batch = yn_batch.cuda()

    loss = model(positive_batch, negative_batch, yp_batch, yn_batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    optimizer.step()

    log = {
        'loss': loss.item()
    }

    return log

def test_step(model, valid_triples, test_dataset_list, entity2id, id2entity, relation2id, id2relation, relation2type, args):
    '''
    Evaluate the model on test or valid datasets
    '''
    model.eval()
    #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
    #Prepare dataloader for evaluation

    logs = []
    detail_logs = [[[],[],[],[]], [[],[],[],[]]]
    dd = ["N-N", "N-1", "1-N", "1-1"]
    mm = {'head':0, 'tail':1}

    step = 0
    total_steps = sum([len(dataset) for dataset in test_dataset_list])

    with torch.no_grad():
        for test_dataset in test_dataset_list:
            for positive_sample, negative_sample, filter_bias, mode in test_dataset:

                if args.cuda:
                    s = "cuda:0"
                    positive_sample = positive_sample.to(s, non_blocking=True)
                    negative_sample = negative_sample.to(s, non_blocking=True)
                    filter_bias = filter_bias.to(s, non_blocking=True)

                batch_size = positive_sample.size(0)

                nentity = negative_sample.size(1)
                score = model.predict(negative_sample.reshape([-1, 3]))
                score = score.reshape([-1, nentity])
                score += filter_bias
                
                argsort = torch.argsort(score, dim = 1, descending=True)
                
                if mode == 'head':
                    positive_arg = positive_sample[:, 0]
                elif mode == 'tail':
                    positive_arg = positive_sample[:, 2]
                else:
                    raise ValueError('mode %s not supported' % mode)

                check_ids = []
                for i in range(batch_size):
                    relation = positive_sample[i][1]
                    #Notice that argsort is not ranking
                    ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                    assert ranking.size(0) == 1

                    #ranking + 1 is the true ranking used in evaluation metrics
                    ranking = 1 + ranking.item()
                    logs.append({
                        'MRR': 1.0/ranking,
                        'MR': float(ranking),
                        'HITS@1': 1.0 if ranking <= 1 else 0.0,
                        'HITS@3': 1.0 if ranking <= 3 else 0.0,
                        'HITS@10': 1.0 if ranking <= 10 else 0.0,
                    })
                    ttype = relation2type[relation.item()]
                    detail_logs[mm[mode]][ttype].append({
                        'MRR_' + dd[ttype] + '_' + mode: 1.0/ranking,
                        'MR_' + dd[ttype] + '_' + mode: float(ranking),
                        'HITS@1_' + dd[ttype] + '_' + mode: 1.0 if ranking <= 1 else 0.0,
                        'HITS@3_' + dd[ttype] + '_' + mode: 1.0 if ranking <= 3 else 0.0,
                        'HITS@10_' + dd[ttype] + '_' + mode: 1.0 if ranking <= 10 else 0.0,
                    })
                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs])/len(logs)
    for i in range(2):
        for j in range(4):
            for metric in detail_logs[i][j][0].keys():
                metrics[metric] = sum([log[metric] for log in detail_logs[i][j]])/len(detail_logs[i][j])
    return metrics