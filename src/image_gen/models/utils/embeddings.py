import torch.nn as nn
from functools import reduce


class embedBlock(nn.Module):
    def __init__(self, input_size: int, layer_sizes: list, act_function=nn.GELU):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.output_size = layer_sizes[-1]
        self.layer_sizes = [
            input_size,
        ] + layer_sizes

        layer_act_pairs = []
        for inp_layer_size, out_layer_size in zip(
            self.layer_sizes[:-1], self.layer_sizes[1:]
        ):
            layer_act_pairs.append(
                [nn.Linear(inp_layer_size, out_layer_size), act_function()]
            )

        # replace activation function in last layer by reshaping
        layer_act_pairs[-1][1] = nn.Unflatten(1, (self.output_size, 1, 1))

        self.embedding = nn.Sequential(*reduce(list.__add__, layer_act_pairs))

    def forward(self, input):
        return self.embedding(input)
