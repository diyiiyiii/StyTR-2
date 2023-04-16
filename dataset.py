import torch.utils.data as data
from pathlib import Path
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        print(self.root)
        self.path = os.listdir(self.root)
        if os.path.isdir(os.path.join(self.root,self.path[0])):
            self.paths = []
            for file_name in os.listdir(self.root):
                for file_name1 in os.listdir(os.path.join(self.root,file_name)):
                    self.paths.append(self.root+"/"+file_name+"/"+file_name1)             
        else:
            self.paths = list(Path(self.root).glob('*'))
        self.transform = transform
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img
    def __len__(self):
        return len(self.paths)
    def name(self):
        return 'FlatFolderDataset'

class RandomTextDataset(data.Dataset):
    def __init__(self, text=['fire', 'pencil', 'water'], prompt_engineering=True):
        super(RandomTextDataset, self).__init__()
        self.text = text
        self.prompt_engineering = prompt_engineering
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def preprocess(self, text):
        if self.prompt_engineering:
            text = "a photo of " + text
        return text
    
    def __getitem__(self, index):
        return self.text[index]
    
    def __len__(self):
        return len(self.text)