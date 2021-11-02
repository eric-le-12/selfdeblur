import torch
import torch.nn as nn


class kernel_approximation():
    def __init__(self, num_input_channels, num_output_channels, num_hidden=1000,mode="normal"):

        model = nn.Sequential()
        model.add_module("Linear_first",
                         nn.Linear(num_input_channels, num_hidden, bias=True))
        model.add_module("activation", nn.ReLU6())
        model.add_module("Linear_second",
                         nn.Linear(num_hidden, num_output_channels, bias=True))
        model.add_module("activation_2", nn.ReLU6())
        model.add_module("prob activation", nn.Softmax())
        self.model = model

    def forward(self, input):
        return self.model(input)
    
    def create_model(self):
        return self.model