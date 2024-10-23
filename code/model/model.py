import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from .matching_layer import MatchingLayer
from transformers.models.t5.modeling_t5 import T5LayerNorm
from einops import rearrange
class Model(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.inference = InferenceLayer(config)
        self.matching = MatchingLayer(config)
        self.interactive = interactivite(config)
        self.interactive_ia = interactivite_ia(config)
        self.ia = nn.Linear(768*2, 768)

    def forward(self, input_ids, attention_mask, ids, length, table_labels_S=None, table_labels_E=None,
                table_labels_iaS=None, table_labels_iaE=None,pairs_true=None,):

        seq_temp = self.bert(input_ids, attention_mask)[0]
        tag = seq_temp[:,seq_temp.size(1) - 2,:]
        last = torch.unsqueeze(seq_temp[:,seq_temp.size(1) - 1,:], 0)
        seq = seq_temp[:,0:length - 1,:]
        seq = torch.cat((seq, last), dim=1)
        sentence_cls = tag.unsqueeze(1).expand([-1, length, -1])

        ia_seq = self.interactive_ia(seq, sentence_cls)

        seq = seq.unsqueeze(2).expand([-1, -1, length, -1])
        seq_T = seq.transpose(1,2)
        table = self.interactive(seq, seq_T)

        output = self.inference(table, attention_mask, table_labels_S, table_labels_E, table_labels_iaS, table_labels_iaE, ia_seq)
        output['ids'] = ids

        output = self.matching(output, table, pairs_true, ia_seq)
        return output


class InferenceLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cls_linear_S = nn.Linear(768, 1)
        self.cls_linear_E = nn.Linear(768, 1)
        self.cls_linear_iaS = nn.Linear(768, 1)
        self.cls_linear_iaE = nn.Linear(768, 1)

    def span_pruning(self, pred, z, attention_mask):
        mask_length = attention_mask.sum(dim=1) - 3
        length = ((attention_mask.sum(dim=1) - 3) * z).long()
        length[length < 10] = 10
        max_length = mask_length ** 2
        for i in range(length.shape[0]):
            if length[i] > max_length[i]:
                length[i] = max_length[i]
        batch_size = attention_mask.shape[0]
        pred_sort, _ = pred.view(batch_size, -1).sort(descending=True)
        batchs = torch.arange(batch_size).to('cuda')
        topkth = pred_sort[batchs, length - 1].unsqueeze(1)
        if topkth[0] == 0:
            return pred > (topkth.view(batch_size, 1, 1))
        else:
            return pred >= (topkth.view(batch_size, 1, 1))

    def forward(self, table, attention_mask, table_labels_S, table_labels_E, table_labels_iaS, table_labels_iaE, ia_seq):
        outputs = {}

        logits_S = torch.squeeze(self.cls_linear_S(table), 3)
        logits_E = torch.squeeze(self.cls_linear_E(table), 3)
        logits_iaS = torch.squeeze(self.cls_linear_iaS(ia_seq), 2)
        logits_iaE = torch.squeeze(self.cls_linear_iaE(ia_seq), 2)
        loss_func1 = nn.BCEWithLogitsLoss(weight=(table_labels_S >= 0))
        loss_func2 = nn.BCEWithLogitsLoss(weight=(table_labels_iaS >= 0))

        outputs['table_loss_S'] = loss_func1(logits_S, table_labels_S.float())
        outputs['table_loss_E'] = loss_func1(logits_E, table_labels_E.float())
        outputs['table_loss_iaS'] = loss_func2(logits_iaS, table_labels_iaS.float())
        outputs['table_loss_iaE'] = loss_func2(logits_iaE, table_labels_iaE.float())

        S_pred = torch.sigmoid(logits_S) * (table_labels_S >= 0)
        E_pred = torch.sigmoid(logits_E) * (table_labels_S >= 0)
        iaS_pred = torch.sigmoid(logits_iaS) * (table_labels_iaS >= 0)
        iaE_pred = torch.sigmoid(logits_iaE) * (table_labels_iaS >= 0)

        if self.config.span_pruning != 0:
            table_predict_S = self.span_pruning(S_pred, self.config.span_pruning, attention_mask)
            table_predict_E = self.span_pruning(E_pred, self.config.span_pruning, attention_mask)
            table_predict_iaS = self.span_pruning(iaS_pred, self.config.span_pruning, attention_mask)
            table_predict_iaE = self.span_pruning(iaE_pred, self.config.span_pruning, attention_mask)
        else:
            table_predict_S = S_pred > 0.5
            table_predict_E = E_pred > 0.5
            table_predict_iaS = iaS_pred > 0.5
            table_predict_iaE = iaE_pred > 0.5

        outputs['table_predict_S'] = table_predict_S
        outputs['table_predict_E'] = table_predict_E
        outputs['table_predict_iaS'] = table_predict_iaS
        outputs['table_predict_iaE'] = table_predict_iaE
        outputs['table_labels_S'] = table_labels_S
        outputs['table_labels_E'] = table_labels_E
        outputs['table_labels_iaS'] = table_labels_iaS
        outputs['table_labels_iaE'] = table_labels_iaE
        return outputs

class interactivite(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_f_1 = nn.Conv2d(
            in_channels=768,
            out_channels=768,
            kernel_size=(3, 3),
            padding=1
        )
        self.conv_f_2 = nn.Conv2d(
            in_channels=768*2,
            out_channels=768*2,
            kernel_size=(3, 3),
            padding=1
        )
        self.conv_b_1 = nn.Conv2d(
            in_channels=768,
            out_channels=768,
            kernel_size=(3, 3),
            padding=1
        )
        self.conv_b_2 = nn.Conv2d(
            in_channels=768*2,
            out_channels=768*2,
            kernel_size=(3, 3),
            padding=1
        )

        self.conv_g_1 = nn.Conv2d(
            in_channels=768,
            out_channels=768,
            kernel_size=(3, 3),
            padding=1,
        )
        self.conv_g_2 = nn.Conv2d(
            in_channels=768,
            out_channels=768,
            kernel_size=(1, 1),
            padding=0,
        )
        self.norm1 = T5LayerNorm(768, 1e-12)
        self.norm2 = T5LayerNorm(768, 1e-12)
        self.W = nn.Linear(768*6, 768)

    def forward(self, seq, seq_T):
        x = rearrange(seq, 'b m n d -> b d m n')
        y = rearrange(seq_T, 'b m n d -> b d m n')

        x_forward = self.conv_f_1(x)
        y_forward = self.conv_f_2(torch.cat((x, y), dim=1))

        y_backward = self.conv_b_1(y)
        x_backward = self.conv_b_2(torch.cat((y, x), dim=1))

        t = torch.cat((x_forward, y_forward, x_backward, y_backward), dim=1)
        tn = rearrange(t, 'b d m n -> b m n d')

        tn = self.W(tn)
        tn = torch.relu(tn)

        tnn = rearrange(tn, 'b m n d-> b d m n ')
        n = x.size(-1)

        tnn = self.conv_g_1(tnn)
        tnn = rearrange(tnn, 'b d m n -> b (m n) d')
        tnn = self.norm1(tnn)
        tnn = torch.relu(tnn)
        tnn = rearrange(tnn, 'b (m n) d -> b d m n', n=n)

        tnn = self.conv_g_2(tnn)
        tnn = rearrange(tnn, 'b d m n -> b (m n) d')
        tnn = self.norm2(tnn)
        tnn = torch.relu(tnn)
        tnn = rearrange(tnn, 'b (m n) d -> b m n d', n=n)
        table = tnn
        return table

class interactivite_ia(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_f_1_ia = nn.Conv1d(
            in_channels=768,
            out_channels=768,
            kernel_size=3,
            padding=1
        )
        self.conv_f_2_ia = nn.Conv1d(
            in_channels=768*2,
            out_channels=768*2,
            kernel_size=3,
            padding=1
        )
        self.conv_b_1_ia = nn.Conv1d(
            in_channels=768,
            out_channels=768,
            kernel_size=3,
            padding=1
        )
        self.conv_b_2_ia = nn.Conv1d(
            in_channels=768*2,
            out_channels=768*2,
            kernel_size=3,
            padding=1
        )

        self.conv_g_1_ia = nn.Conv1d(
            in_channels=768,
            out_channels=768,
            kernel_size=3,
            padding=1,
        )
        self.conv_g_2_ia = nn.Conv1d(
            in_channels=768,
            out_channels=768,
            kernel_size=1,
            padding=0,
        )
        self.norm1_ia = T5LayerNorm(768, 1e-12)
        self.norm2_ia = T5LayerNorm(768, 1e-12)
        self.W_ia = nn.Linear(768 * 6, 768)

    def forward(self, seq, seq_T):
        x = rearrange(seq, 'b m d -> b d m')
        y = rearrange(seq_T, 'b m d -> b d m')

        x_forward = self.conv_f_1_ia(x)
        y_forward = self.conv_f_2_ia(torch.cat((x, y), dim=1))

        y_backward = self.conv_b_1_ia(y)
        x_backward = self.conv_f_2_ia(torch.cat((y, x), dim=1))

        tn = torch.cat((x_forward, y_forward, x_backward, y_backward), dim=1)
        tn = rearrange(tn, 'b d m-> b m d')

        tn = self.W_ia(tn)
        tn = torch.relu(tn)
        tnn = rearrange(tn, 'b m d-> b d m')

        tnn = self.conv_g_1_ia(tnn)
        tnn = rearrange(tnn, 'b d m -> b m d')
        tnn = self.norm1_ia(tnn)
        tnn = torch.relu(tnn)
        tnn = rearrange(tnn, 'b m d-> b d m')

        tnn = self.conv_g_2_ia(tnn)
        tnn = rearrange(tnn, 'b d m -> b m d')
        tnn = self.norm2_ia(tnn)
        tnn = torch.relu(tnn)
        table = tnn
        return table