import torch
import cv2
import numpy as np
from torch.utils.data import Dataset


class WheatDataset(Dataset):
    def __init__(self, dataframe, image_dir):
        super().__init__()
        self.image_ids = dataframe["image_id"].unique()
        self.image_dir = image_dir
        self.df = dataframe

    def __len__(self):
        return self.image_ids.shape[0]
    
    def __getitem__(self, item):
        image_id = self.image_ids[item]
        records = self.df[self.df["image_id"] == image_id]
        
        image = cv2.imread(f"{self.image_dir}/{image_id}.jpg", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        boxes = records[['x', 'y', 'w', 'h']].values
        area = (boxes[:, 2] * boxes[:, 3])
        area = torch.tensor(area, dtype=torch.float32)

        # There is only one class (the wheet)
        labels = torch.ones((records.shape[0],), dtype=torch.int64)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0], 1), dtype=torch.int64)
        
        # target must have boxes coordinates, labels, image_id, area, iscrowd
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([item])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
  
        return image, target, image_id

