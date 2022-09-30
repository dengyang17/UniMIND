from transformers import BartForConditionalGeneration
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F


def init_params(model):
    for name, param in model.named_parameters():
        if param.data.dim() > 1:
            xavier_uniform_(param.data)
        else:
            pass

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, din):
        dout = F.relu(self.fc1(din))
        dout = F.relu(self.fc2(dout))
        return dout


class UniMind(nn.Module):
    def __init__(self, args, config, item_num):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                    config=config, cache_dir=args.cache_dir)
        d_model = config.d_model
        self.mlp = MLP(d_model, d_model//4, d_model//2)
        self.classifier = nn.Linear(d_model//4, item_num)
        #self.criterion = F.binary_cross_entropy_with_logits
        self.criterion = F.cross_entropy
        self.item_num = item_num
        self.config = config
        init_params(self.classifier)
        init_params(self.mlp)

    def forward(self, input_ids, attention_mask, hist_ids=None, labels=None, item_ids=None):
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        encode_state = outputs.decoder_hidden_states[-1][:,0,:].squeeze(1)
        hidden = self.mlp(encode_state)
        res = self.classifier(hidden)
        if self.training:
            loss_g = outputs[0]
            #loss_r = self.criterion(res, F.one_hot(item_ids, self.item_num).squeeze(1).float())
            loss_r = self.criterion(res, item_ids.squeeze(1), ignore_index=self.item_num-1)
            return loss_g + loss_r, res, loss_g
        return res