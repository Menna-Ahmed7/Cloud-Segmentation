import os
from segmentation_models_pytorch import Unet
from skimage.transform import resize 
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
import pandas as pd
import tifffile as tiff
test_folder = "D:/ComputerEngineering/Fourth_Year/Second Term/Satellites/New folder/Cloud-Segmentation/archive/test/test/data"  # or your test images path
submission_sample_path="D:/ComputerEngineering/Fourth_Year/Second Term/Satellites/New folder/Cloud-Segmentation/archive/sample_submission.csv"
model_path="D:/ComputerEngineering/Fourth_Year/Second Term/Satellites/New folder/Cloud-Segmentation/new/unet_epochs30_lr0.0005_pretrained_densenet_lr_dice_loss_with_threshold/best_model (3).pth"
# ---------Encoder---------
def rle_encode(mask):
    """
    Encodes a binary mask using Run-Length Encoding (RLE).    
    Args:
        mask (np.ndarray): 2D binary mask (0s and 1s).
    Returns:
        str: RLE-encoded string, or a single space " " if mask is all zeros.
    """
    if np.sum(mask) == 0:
        return " "  # As it seems that kaggle reject nulls. We'll handle cloud-free images with empty spaces.
    
    pixels = mask.flatten(order='F')  # Flatten in column-major order
    pixels = np.concatenate([[0], pixels, [0]])  # Add padding to detect transitions
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1  # Get transition indices
    runs[1::2] -= runs[::2]  # Compute run lengths
    runs[::2] -= 1  # Make it 0-indexed instead of 1-indexed

    return " ".join(map(str, runs))  # Convert to string format
# -----Data Preparation-----------
class TestCloudSegmentationDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image using tifffile
        img_array = tiff.imread(self.image_paths[idx])  # Shape: (H, W, bands) or (bands, H, W)

        # Ensure bands-first shape (bands, H, W)
        if img_array.ndim == 3 and img_array.shape[0] != 2:
            if img_array.shape[-1] >= 4:
                img_array = np.moveaxis(img_array, -1, 0)  # Convert (H, W, bands) → (bands, H, W)
            else:
                raise ValueError(f"Image at {self.image_paths[idx]} has less than 4 bands")

        # Select bands 1 and 4 → (2, H, W)
        selected_bands = np.stack([img_array[0], img_array[3]], axis=0)

        # Resize each band to (256, 256)
        resized_bands = np.zeros((2, 256, 256), dtype=np.float32)
        for i in range(2):
            resized_bands[i] = resize(selected_bands[i], (256, 256), order=1, preserve_range=True, anti_aliasing=True)

        # Convert to tensor
        image_tensor = torch.tensor(resized_bands, dtype=torch.float32)

        return image_tensor, os.path.basename(self.image_paths[idx])


# --------------Main--------------
# 1. Define the path to your test folder

# 2. Get all image paths from the test folder
def get_image_paths(folder):
    """Get paths of all image files in a folder"""
    valid_extensions = ('.tif', '.tiff')
    return [os.path.join(folder, f) for f in os.listdir(folder) 
            if f.lower().endswith(valid_extensions)]


def process_model_outputs(test_loader, model, output_csv_path='team_12.csv'):
    # Prepare CSV file
    threshold=0.7
    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['id', 'segmentation'])  # Write header
        
        # Process each batch
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            for images, image_names in test_loader:
                images = images.to(device)  
                # Get model predictions
                outputs = model(images)
                 # Get predicted probabilities 
                probs = torch.softmax(outputs, dim=1)
                
                # Get foreground probabilities (class 1)
                fg_probs = probs[:, 1]
                
                # Apply threshold to get binary predictions
                preds = (fg_probs > threshold).float()
        
                # masked_predictions = (masked_predictions == 1).cpu().numpy().astype(np.uint8)
                # Process each prediction in the batch
                for i in range(preds.shape[0]):
                    #### REsise mask to 256, 256####
                    pred = preds[i].cpu().numpy()
                    # pred = resize(pred, (256, 256), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
                    rle = rle_encode(pred)
                    image_name = image_names[i]
                    image_name = os.path.splitext(image_names[i])[0]                    
                    # Write to CSV
                    csv_writer.writerow([image_name, rle])
    
    print(f"Results saved to {output_csv_path}")
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sample_df = pd.read_csv(submission_sample_path, dtype={'id': str})
# Ensure ordered image paths
ordered_filenames = sample_df['id'].tolist()
# Append '.tif' to match actual file names
ordered_filenames_with_ext = [f"{fname}.tif" for fname in ordered_filenames]
test_image_paths_dict = {os.path.basename(p): p for p in get_image_paths(test_folder)}
# Retrieve full paths
ordered_image_paths = [test_image_paths_dict[fname] for fname in ordered_filenames_with_ext if fname in test_image_paths_dict]

# Create test dataset and dataloader
test_dataset = TestCloudSegmentationDataset(ordered_image_paths)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

print(f"{len(test_dataset)} testing samples")

# Model setup (on CPU)
device = torch.device("cpu")  # Force CPU
model = Unet(encoder_name="densenet121", in_channels=2, classes=2).to(device)

# Don't wrap with DataParallel on CPU
# Load model parameters with map_location for CPU
state_dict=torch.load(model_path, map_location=device)
# Strip "module." from keys if present (i.e., saved from DataParallel)
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("module.", "") if k.startswith("module.") else k
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)

process_model_outputs(test_loader, model)

# Load generated and reference submission files
generated_df = pd.read_csv("team_12.csv", dtype={'id': str})
reference_df = pd.read_csv(submission_sample_path, dtype={'id': str})

# Assert that both have the same IDs in the same order
assert list(generated_df['id']) == list(reference_df['id']), "ID order mismatch between output and sample submission."


print("ID order in output CSV matches the sample submission.")