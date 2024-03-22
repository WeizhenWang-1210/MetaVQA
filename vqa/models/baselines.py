import torch
import torch.nn as nn


class Baseline(nn.Module):
    """
    The simplest architecture such that the features are concatenated together and put thourgh a predictor network.
    """

    def __init__(self, text_encoder, rgb_encoder, predictor, name):
        super(Baseline, self).__init__()
        self.text_encoder = text_encoder
        self.rgb_encoder = rgb_encoder
        self.predictor = predictor
        self.name = name

    def forward(self, question, mask, img, lidar):
        img_feature = self.rgb_encoder(img)
        text_feature = self.text_encoder(question, mask)
        nn_input = torch.cat((img_feature, text_feature, lidar), dim=1)
        return self.predictor(nn_input)
