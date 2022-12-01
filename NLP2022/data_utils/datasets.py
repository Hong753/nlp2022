import os
import random
import pickle
import pandas as pd
import numpy as np

from nltk.tokenize import RegexpTokenizer
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CUBDataset(Dataset):
    def __init__(self, 
                 data_path, 
                 num_captions=10, 
                 num_words=18, 
                 img_resolution=256, 
                 split="train"):
        self.data_path = data_path
        self.num_captions = num_captions
        self.num_words = num_words
        self.transforms = transforms.Compose([transforms.Resize(img_resolution * 76 // 64), # 256 * 76 / 64
                                              transforms.RandomCrop(img_resolution), # original image size
                                              transforms.RandomHorizontalFlip(p=0.5), 
                                              transforms.ToTensor(), 
                                              transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        
        # Load bounding boxes for all data
        self.bbox = self.load_bbox()
        
        # Load filenames for train/test data
        self.filenames = self.load_filenames(split=split)
        
        # Load class ids for train/test data
        self.class_ids = self.load_class_ids(split=split)
        
        # Load captions for train/test data
        self.captions = self.load_captions(filenames=self.filenames, split=split)
        
        # Construct mapping dictionary
        self.captions_id, self.id2word, self.word2id = self.construct_vocabulary(split=split)
    
    def load_bbox(self):
        # Get bounding boxes
        bbox_path = os.path.join(self.data_path, "bounding_boxes.txt")
        assert os.path.exists(bbox_path), "Check for 'bounding_boxes.txt' in {}".format(self.data_path)
        df_bounding_boxes = pd.read_csv(bbox_path, delim_whitespace=True, header=None).astype(int)
        
        # Get filenames
        file_path = os.path.join(self.data_path, "images.txt")
        assert os.path.exists(file_path), "Check for 'images.txt' in {}".format(self.data_path)
        df_filenames = pd.read_csv(file_path, delim_whitespace=True, header=None)
        
        # Construct dictionary (filename: bbox)
        filenames = df_filenames[1].tolist()
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        for idx in range(len(filenames)):
            bbox = df_bounding_boxes.iloc[idx][1:].tolist()
            key = filenames[idx][:-4]
            filename_bbox[key] = bbox
        return filename_bbox
        
    def load_filenames(self, split="train"):
        assert split in ["train", "test"]
        file_path = os.path.join(self.data_path, split, "filenames.pickle")
        assert os.path.exists(file_path), "Check for 'filenames.pickle' in {}".format(os.path.join(self.data_path, split))
        with open(file_path, "rb") as f:
            filenames = pickle.load(f)
        return np.asarray(filenames)
    
    def load_class_ids(self, split="train"):
        assert split in ["train", "test"]
        file_path = os.path.join(self.data_path, split, "class_info.pickle")
        assert os.path.exists(file_path), "Check for 'class_info.pickle' in {}".format(os.path.join(self.data_path, split))
        with open(file_path, "rb") as f:
            filenames = pickle.load(f, encoding="latin1")
        return np.asarray(filenames)
    
    def load_captions(self, filenames, split="train"):
        assert split in ["train", "test"]
        all_captions = []
        for idx in range(len(filenames)):
            caption_path = os.path.join(self.data_path, "text", f"{filenames[idx]}.txt")
            with open(caption_path, "r") as f:
                captions = f.read().encode("utf-8").decode("utf8").split("\n")
                count = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ") # pick sequences of alphanumeric characters as tokens
                    tokenizer = RegexpTokenizer(r"\w+")
                    tokens = tokenizer.tokenize(cap.lower())
                    if len(tokens) == 0:
                        print("cap", cap)
                        continue
                    
                    new_tokens = []
                    for t in tokens:
                        t = t.encode("ascii", "ignore").decode("ascii")
                        if len(t) > 0:
                            new_tokens.append(t)
                    all_captions.append(new_tokens)
                    count += 1
                    if count == self.num_captions:
                        break
                if count < self.num_captions:
                    print("The number of captions for {} is {} < 10.".format(filenames[idx], count))                    
        return all_captions
    
    def construct_vocabulary(self, split="train"):
        assert split in ["train", "test"]
        file_path = os.path.join(self.data_path, "captions_DAMSM.pickle")
        assert os.path.exists(file_path), "Check for 'captions_DAMSM.pickle' in {}".format(self.data_path)
        with open(file_path, "rb") as f:
            train_captions_id, test_captions_id, id2word, word2id = pickle.load(f)
        if split == "train":
            captions_id = train_captions_id
        else:
            captions_id = test_captions_id
        return captions_id, id2word, word2id
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        # class_id = self.class_id[idx]
        bbox = self.bbox[filename]
        image_path = os.path.join(self.data_path, "images", f"{filename}.jpg")
        image = self._preprocess_image(image_path=image_path, bbox=bbox, transforms=self.transforms)
        caption_idx = idx * self.num_captions + random.randint(0, self.num_captions - 1)
        caption, caption_length = self._get_caption(caption_idx)
        return {"images": image, 
                "captions": caption, 
                "caption_lengths": caption_length}
    
    def __len__(self):
        return len(self.filenames)
    
    @staticmethod
    def _preprocess_image(image_path, bbox=None, transforms=None):
        image = Image.open(image_path).convert("RGB")
        W, H = image.size
        if bbox is not None:
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(H, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(W, center_x + r)
            image = image.crop([x1, y1, x2, y2])
        if transforms is not None:
            image = transforms(image)
        return image
    
    def _get_caption(self, caption_idx):
        caption = np.asarray(self.captions_id[caption_idx]).astype("int64")
        caption_length = len(caption)
        out = np.zeros((self.num_words, ), dtype="int64")
        if caption_length <= self.num_words:
            out[:caption_length] = caption
        else:
            idx = list(np.arange(caption_length))
            np.random.shuffle(idx)
            idx = idx[:self.num_words]
            idx = np.sort(idx) # TODO: why sort?
            out[:] = caption[idx]
            caption_length = self.num_words
        return out, caption_length