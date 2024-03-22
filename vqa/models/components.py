import torch
from transformers import BertModel
import torch.nn as nn
import torchvision.models as models


class MLP_Multilabel(nn.Module):
    """
    Multi-level-perceptron with variable hidden dim and variable number of hidden dims.
    Note that we are doing multi_label classfication, the last layer is a position-wise
    sigmoid instead of softmax
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden) -> None:
        super(MLP_Multilabel, self).__init__()
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
        self.network = self.create_network()

    def forward(self, x):
        return self.network(x)

    def create_network(self):
        layers = [nn.Linear(self.in_dim, self.hidden_dim), nn.ReLU()]
        for _ in range(self.num_hidden):
            layers += [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()]
        layers += [nn.Linear(self.hidden_dim, self.out_dim)]
        network = nn.Sequential(
            *layers,
            torch.nn.Sigmoid()
        )
        return network


class Bert_Encoder(nn.Module):
    """
    Given a tokenized sequence, return the embedding vector with averaging.
    The parameters are frozen.
    """

    def __init__(self):
        super(Bert_Encoder, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def forward(self, text, mask):
        with torch.no_grad():
            outputs = self.model(text, mask)
            last_hidden_states = outputs[0]
            text_feature = last_hidden_states.mean(axis=1)
        return text_feature


class Resnet50_Encoder(nn.Module):
    """
    Given an Imagenet transformed image tensor, return the embedding vector after the final average pooling.
    The parameters are frozen.
    """

    def __init__(self, weight_id="ResNet50_Weights.IMAGENET1K_V1"):
        super(Resnet50_Encoder, self).__init__()
        self.weight_id = weight_id
        self.network = self.setup(weight_id)

    @classmethod
    def setup(cls, weight):
        model = models.resnet50(weights=weight)
        model.eval()
        modules = list(model.children())[:-1]
        encoder = nn.Sequential(*modules)
        for p in encoder.parameters():
            p.requires_grad = False
        return encoder

    def forward(self, x):
        return self.network(x).squeeze(-1).squeeze(-1)


