import torch
import torch.nn as nn
from .crf import CRF
from .transformers.modeling_bert import BertPreTrainedModel
from .transformers.modeling_bert import BertModel
from .layers import Attention
from torch.nn import LayerNorm

class BertCrfForAttr(BertPreTrainedModel):
    def __init__(self, config, label2id, device):
        super(BertCrfForAttr, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.t_lstm = nn.LSTM(input_size=config.hidden_size,hidden_size=config.hidden_size // 2,
                                batch_first=True,bidirectional=True)
        self.a_lstm = nn.LSTM(input_size=config.hidden_size,hidden_size=config.hidden_size // 2,
                                batch_first=True,bidirectional=True)
        self.attention = Attention()
        self.ln = LayerNorm(config.hidden_size* 2)
        self.classifier = nn.Linear(config.hidden_size* 2, len(label2id))
        self.crf = CRF(tagset_size=len(label2id), tag_dictionary=label2id, device=device, is_bert=True)
        self.init_weights()

    def forward(self, t_input_ids,a_input_ids, t_token_type_ids=None, t_attention_mask=None,
                 a_token_type_ids=None, a_attention_mask=None,labels=None,input_lens=None):
        # bert
        outputs_title = self.bert(t_input_ids, t_token_type_ids, t_attention_mask)
        outputs_attr = self.bert(a_input_ids, a_token_type_ids, a_attention_mask)
        # bilstm
        title_output, _ = self.t_lstm(outputs_title[0])
        _, attr_hidden = self.a_lstm(outputs_attr[0])
        # attention
        attr_output = torch.cat([attr_hidden[0][-2],attr_hidden[0][-1]], dim=-1)
        attention_output = self.attention(title_output,attr_output)
        # catconate
        outputs = torch.cat([title_output, attention_output], dim=-1)
        outputs = self.ln(outputs)
        sequence_output = self.dropout(outputs)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf.calculate_loss(logits, tag_list=labels, lengths=input_lens)
            outputs =(loss,)+outputs
        return outputs # (loss), scores


