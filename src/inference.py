import torch
import pandas as pd
import numpy as np
import re

import config
from utils import collate_fn
from utils import get_test_transform
from utils import format_prediction_string
from model import load_model
from dataset import WheatDatasetTest

def predict(test_path):
    test_df = pd.read_csv(test_path)
    
    # split the data 
    image_ids = test_df['image_id'].unique()
    
    test_dataset = WheatDatasetTest(test_df, config.DIR_TEST, get_test_transform())
    
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST_BS,
        shuffle=False,
        drop_last=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn
    )

    device = torch.device('cuda')
    detection_threshold = 0.5
    results = []

    # load the model
    model = load_model(config.WEIGHTS_FILE)
    model.to(device)
    model.eval()

    for images, image_ids in test_data_loader:
        images = list(image.to(device) for image in images)
        outputs = model(images)

        for i, image in enumerate(images):
            boxes = outputs[i]['boxes'].data.cpu().numpy()
            scores = outputs[i]['scores'].data.cpu().numpy()
            
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            scores = scores[scores >= detection_threshold]
            image_id = image_ids[i]
            
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            
            result = {
                'image_id': image_id,
                'PredictionString': format_prediction_string(boxes, scores)
            }
            results.append(result)
    
    test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
    test_df.to_csv('../output/output.csv')
    
    return test_df


if __name__ == "__main__":
    test_path = "../Dataset/sample_submission.csv"
    predictions = predict(test_path)
    print(predictions)
