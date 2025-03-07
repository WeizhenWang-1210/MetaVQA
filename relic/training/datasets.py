import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset
import json
from torchvision import transforms
from PIL import Image
import os
from collections import defaultdict


class MultiChoiceDataset(Dataset):
    """
    Close-vocabulary dataset for MetaVQA. The task of VQA is reduced to Multi-label prediction, trained with
    nn.BCELoss().
    The image transformation follows ImageNet transformations and then images from different perspective
    are naively concatenated The text tokenization utilized BERT tokenizer with maximum sequence length of 64,
    with special symbol paddings. Attention masks are also calculated.
    """

    def __init__(self, qa_paths, split, indices, map,
                 base = "./",
                 img_transform = transforms.Compose([
                                    transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ]),
                 ) -> None:
        super(MultiChoiceDataset, self).__init__()
        try:
            with open(qa_paths, "r") as f:
                qa = json.load(f)
        except Exception as E:
            raise E
        self.split = split
        assert self.split in ["train", "val", "test"]
        self.indices = indices
        self.data = self.load_split(qa, self.indices)
        self.answer_dim = qa["answer_space"]
        self.img_transform = img_transform
        self.text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.map = map
        self.base = base

    @classmethod
    def load_split(cls, qa, indices):
        data = {}
        for idx, i in enumerate(indices):
            data[idx] = qa["qas"][i]
        return data

    def __len__(self):
        return len(self.data)

    def encode_rgb(self, image_paths, multiview=False):
        buffer = []
        for angle, paths in image_paths.items():
            for path in paths:
                rebased = os.path.join(self.base, path)
                img = Image.open(rebased)
                transformed = self.img_transform(img)
                buffer.append(transformed)

        return torch.stack(buffer, dim=0)


    def encode_lidar(self, lidar_paths):
        """
        Return the lidar directly as concatenated tensor
        """
        buffer = []
        for lidar in lidar_paths:
            lidar = os.path.join(self.base, lidar)
            with open(lidar, "r") as file:
                data = json.load(file)
            feature = torch.tensor(data["lidar"])*50 #multiplied with 50 to restore into world scale.
            buffer.append(feature)
        return torch.stack(buffer, dim=0)

    def encode_text(self, text):
        tokens = self.text_tokenizer.encode(text, add_special_tokens=True, padding="max_length", max_length=64)
        mask = [1 if token != 0 else 0 for token in tokens]
        ttensor, mtensor = torch.tensor(tokens), torch.tensor(mask)
        return ttensor, mtensor

    def __getitem__(self, index):
        #index = f"{index}"
        question, image_paths, lidar_paths, answers = self.data[index]["question"], self.data[index]["rgb"], \
            self.data[index]["lidar"], self.data[index]["answer"]
        gt = torch.zeros(5203)
        assert isinstance(answers, list)
        for i in answers:
            gt[i] = 1
        question_z, question_mask = self.encode_text(question)
        image_z = self.encode_rgb(image_paths)
        lidar_z = self.encode_lidar([lidar_paths])
        # print(question_z.shape, question_mask.shape,image_z.shape, lidar_z.shape, gt.shape)
        return question_z, question_mask, image_z.squeeze(), lidar_z.squeeze(), gt, index
