from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import pandas as pd
import json

def get_files_with_extension(root_dir, target_extension):
    file_list = []

    def process_directory(root, files):
        try:
            return [os.path.abspath(os.path.join(root, file_name)) for file_name in files if file_name.endswith(target_extension)]
        except Exception as e:
            print(f"Error processing directory {root}: {e}")
            return []

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_directory, root, files): root for root, _, files in os.walk(root_dir)}
        
        for future in as_completed(futures):
            try:
                file_list.extend(future.result())
            except Exception as e:
                root = futures[future]
                print(f"Error processing future for directory {root}: {e}")

    return file_list

def get_files_with_extension_generator(root_dir, target_extension):
    def process_directory(root, files):
        try:
            return (os.path.abspath(os.path.join(root, file_name)) for file_name in files if file_name.endswith(target_extension))
        except Exception as e:
            print(f"Error processing directory {root}: {e}")
            return []

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_directory, root, files): root for root, _, files in os.walk(root_dir)}
        
        for future in as_completed(futures):
            try:
                for file_path in future.result():
                    yield file_path
            except Exception as e:
                root = futures[future]
                print(f"Error processing future for directory {root}: {e}")


class MultiCamDataset(Dataset):
    def __init__(self, annotation_path, transform):
        # self.pd_data = pd.read_csv(annotation_path, header=None)
        self.json_data = json.load(open(annotation_path))
        # self.image_list = get_files_with_extension(root_dir, '.jpg')
        # self.image_list = file_list
        
        self.transform = transform

    def __len__(self):
        return len(self.json_data)
    
    def __getitem__(self, idx):
        # image_path = self.image_list[idx]
        # image_path = self.pd_data.iloc[idx, 0]
        image_path = self.json_data[str(idx)]
        # image = Image.open(image_path)
        image = read_image(image_path)
        image = transforms.ToPILImage()(image)
        
        if self.transform:
            image = self.transform(image)
        
        label = image_path.split('/')[-1].split('_')[0]
        label = 0 if label == 's' else 1 # ['silent', 'utter']

        return image, label


# class MultiCamDataset(Dataset):
#     def __init__(self, root_dir, transform):
#         self.image_list = list(get_files_with_extension_generator(root_dir, '.jpg'))
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_list)
    
#     def __getitem__(self, idx):
#         image_path = self.image_list[idx]
#         image = Image.open(image_path).convert("RGB")
        
#         if self.transform:
#             image = self.transform(image)
        
#         label = image_path.split('/')[-1].split('_')[0]
#         label = 0 if label == 's' else 1  # ['silent', 'utter']

#         return image, label