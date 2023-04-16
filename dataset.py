import torch.utils.data as data
from pathlib import Path
import os
import torch
import torchvision
from PIL import Image
from transformers import AutoTokenizer, CLIPTextModel, CLIPImageProcessor, CLIPVisionModel
from template import imagenet_templates

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

# def collate_fn_image(batch):
#     preprocessor = CLIPImageProcessor()
#     batch = preprocessor(batch)
#     model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
#     batch = model.encode_image(**batch)
#     return batch

class ImageTokenDataset(data.Dataset):
    '''
    Dataset that uses clip image encoder to encode image into tokens
    '''
    def __init__(self, image_dir: Path):
        super(ImageTokenDataset, self).__init__()
        self.image_dir = image_dir
        self.image_processor = CLIPImageProcessor()
        self.image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.image_embedding = dict()

        self.images = [f for f in self.image_dir.glob('*')]

    def __getitem__(self, index):
        # use preprocessor to break image into patches
        # and use vision model to encode patches into tokens
        # TODO Move this part to model 
        img = torchvision.io.read_image(str(self.images[index]))
        img = self.image_processor(img)
        img['pixel_values'] = torch.tensor(img['pixel_values'])
        img = self.image_encoder(**img)
        return img.last_hidden_state.squeeze(0)

    def __len__(self):
        return len(self.images)


class RandomTextDataset(data.Dataset):
    '''
    Dataset that returns random text descript for style transfer
    '''
    def __init__(self, text=['fire', 'pencil', 'water'], prompt_engineering=True):
        super(RandomTextDataset, self).__init__()
        self.text = text
        self.prompt_engineering = prompt_engineering
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.text_embedding = dict() # storing the 3 kinds of text embedding for each text
        self.index_to_text = dict()

        self.preprocess_text(text)
    
    def compose_text_with_templates(self, text: str, templates=imagenet_templates) -> list:
        return [template.format(text) for template in templates]

    def preprocess_text(self, text: list[str]) -> dict:
        for idx, style_description in enumerate(text):
            self.index_to_text[idx] = style_description
            if not self.prompt_engineering:
                template_text = ["a photo of " + text]
            else:
                template_text = self.compose_text_with_templates(style_description, imagenet_templates)
                embedding_dict = dict()
                with torch.no_grad():
                    inputs = self.tokenizer(template_text, padding=True, return_tensors="pt")
                    outputs = self.text_encoder(**inputs)
                    last_hidden_state = outputs['last_hidden_state']
                    cls_token = outputs['pooler_output']
                    # two option, one is use cls token for encoder
                    # the other one is use average pooling over hidden state
                    # we are gonna do all of them!
                    # option 1 cls token
                    embedding_dict['cls_token'] = torch.mean(cls_token, dim=0)
                    # option 2 global average pooling
                    embedding_dict['average_pooling'] = torch.mean(last_hidden_state, dim=(0,1))
            self.text_embedding[style_description] = embedding_dict
    
    def __getitem__(self, index):
        return self.text_embedding[self.index_to_text[index]]
    
    def __len__(self):
        return len(self.text_embedding)

def test_text_encoder():
    "https://discuss.huggingface.co/t/last-hidden-state-vs-pooler-output-in-clipvisionmodel/26281"
    "average pooling is better than last hidden state???"
    "why not use both?"
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    input_prompts = [template.format("pencil style") for template in imagenet_templates]
    inputs = tokenizer(input_prompts, padding=True, return_tensors="pt")
    outputs = text_encoder(**inputs)

    print(len(imagenet_templates))
    print(outputs['last_hidden_state'].shape)


def test_text_loader():
    text_dataset = RandomTextDataset()
    print(text_dataset[0]['cls_token'].shape)
    print(text_dataset[0]['average_pooling'].shape)

def test_image_loader():
    image_dataset = ImageTokenDataset(Path("input_content/"))
    image_dataloader = data.DataLoader(image_dataset, batch_size=4, shuffle=False, num_workers=0)
    for i, input in enumerate(image_dataloader):
        print(input)
        break

def test_image_encoder():
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPImageProcessor()
    img = torch.rand(4, 3, 224, 224)
    img = processor(img)
    img['pixel_values'] = torch.tensor(img['pixel_values'])
    output = model(**img)
    print(output.keys())

#test_text_encoder()
#test_text_loader()
#test_image_encoder()
test_image_loader()