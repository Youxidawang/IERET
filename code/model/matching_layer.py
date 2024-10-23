import torch
from torch import nn
from torch.nn import functional as F


class MatchingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(768 * 3, 4)

    def gene_pred(self, batch_size, S_preds, E_preds, iaS_preds, iaE_preds, pairs_true):
        all_pred = [[] for i in range(batch_size)]
        pred_label = [[] for i in range(batch_size)]
        pred_maxlen = 0
        for i in range(batch_size):
            S_pred = torch.nonzero(S_preds[i]).cpu().numpy()
            E_pred = torch.nonzero(E_preds[i]).cpu().numpy()
            iaS_pred = torch.nonzero(iaS_preds[i]).cpu().numpy()
            iaE_pred = torch.nonzero(iaE_preds[i]).cpu().numpy()
            for (_,ias) in iaS_pred:
                for (_,iae) in iaE_pred:
                        if ias <= iae:
                            sentiment = 0
                            for j in range(len(pairs_true[i])):
                                p = pairs_true[i][j]
                                if [-1, -1, ias - 1, iae] == p[:4]:
                                    sentiment = p[4]
                            pred_label[i].append(sentiment)
                            all_pred[i].append([-1, -1, ias - 1, iae])
            for (s0, s1) in S_pred:
                for (e0, e1) in E_pred:
                    if s0 <= e0 and s1 <= e1:
                        sentiment = 0
                        for j in range(len(pairs_true[i])):
                            p = pairs_true[i][j]
                            if [s0 - 1, e0, s1 - 1, e1] == p[:4]:
                                sentiment = p[4]
                        pred_label[i].append(sentiment)
                        all_pred[i].append([s0 - 1, e0, s1 - 1, e1])
            if len(all_pred[i]) > pred_maxlen:
                pred_maxlen = len(all_pred[i])
        for i in range(batch_size):
            for j in range(len(all_pred[i]), pred_maxlen):
                pred_label[i].append(-1)
        pred_label = torch.tensor(pred_label).to('cuda')
        return all_pred, pred_label, pred_maxlen

    def input_encoding(self, batch_size, pairs, maxlen, table, ia_seq):
        input_ret = torch.zeros([batch_size, maxlen, 768 * 3]).to('cuda')
        for i in range(batch_size):
            j = 0
            for (s0, e0, s1, e1) in pairs[i]:
                if s0 != -1:
                    S = table[i, s0 + 1, s1 + 1, :]
                    E = table[i, e0, e1, :]
                    R = torch.max(torch.max(table[i, s0 + 1:e0 + 1, s1 + 1:e1 + 1, :], dim=1)[0], dim=0)[0]
                    input_ret[i, j, :] = torch.cat([S,E,R])
                    j += 1
                else:
                    S = ia_seq[i, s1 + 1, :]
                    E = ia_seq[i, e1, :]
                    R = torch.max(ia_seq[i, s1 + 1:e1 + 1, :], dim=0)[0]
                    input_ret[i, j, :] = torch.cat([S,E,R])
                    j += 1
        return input_ret

    def forward(self, outputs, Table, pairs_true, ia_seq):
        table = Table.clone()
        batch_size = table.size(0)
        all_pred, pred_label, pred_maxlen = self.gene_pred(batch_size, outputs['table_predict_S'], outputs['table_predict_E'], outputs['table_predict_iaS'], outputs['table_predict_iaE'], pairs_true)
        pred_input = self.input_encoding(batch_size, all_pred, pred_maxlen, table, ia_seq)
        pred_output = self.linear(pred_input)
        loss_func = nn.CrossEntropyLoss(ignore_index=-1)
        loss_input = pred_output
        loss_label = pred_label
        if loss_input.shape[1] == 0:
            loss_input = torch.zeros([batch_size, 1, 2])
            loss_label = torch.zeros([batch_size, 1]) - 1
        outputs['pair_loss'] = loss_func(loss_input.transpose(1, 2), loss_label.long())
        pairs_logits = F.softmax(pred_output, dim=2)
        if pairs_logits.shape[1] == 0:
            outputs['pairs_preds'] = []
            return outputs

        pairs_pred = pairs_logits.argmax(dim=2)

        outputs['pairs_preds'] = []
        for i in range(batch_size):
            # torch.cuda.empty_cache()
            for j in range(len(all_pred[i])):
                if pairs_pred[i][j] >= 1:
                    se = all_pred[i][j]
                    outputs['pairs_preds'].append((i, se[0], se[1], se[2], se[3], pairs_pred[i][j].item()))

        return outputs

