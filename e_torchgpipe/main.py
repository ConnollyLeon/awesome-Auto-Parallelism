'''
An example of using torchgpipe.

torchgpipe only support model implemented by nn.Sequential.

'''

import torch

from transformers import BertConfig, AdamW, BertLayer
from torchgpipe import GPipe
from torchgpipe.balance import balance_by_time

if __name__ == '__main__':
    # BertLarge
    configuration = BertConfig(hidden_size=1024, num_hidden_layers=24,
                               num_attention_heads=16, intermediate_size=4096)

    # Creating a Sequential Model
    l1 = BertLayer(configuration)
    l2 = BertLayer(configuration)
    l3 = BertLayer(configuration)
    l4 = BertLayer(configuration)

    model = torch.nn.Sequential(l1, l2, l3, l4)

    optimizer = AdamW(model.parameters())

    # Get device_numbers
    partitions = torch.cuda.device_count()

    sample = torch.randint(1, 30000, (1, 512), dtype=torch.long)

    # balance the partitions  by time.
    balance = balance_by_time(partitions, model, sample)

    model = GPipe(model, balance, chunks=8)

    output = model(sample)
