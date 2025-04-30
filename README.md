# Cloud Segmentation for Satellite Imagery

This repository contains the implementation of a cloud segmentation project, developed as a part of sattelite imagery subject at the Computer Engineering Department at the Faculty of Engineering. The project focuses on detecting clouds in satellite images using classical and deep learning methods, achieving a public leaderboard score of 0.79 and a private score of 0.77, ranking 8th.

## Project Overview

The goal of this project is to segment clouds in satellite imagery we tried a classical machine learning algorithm (Random Forest) and deep learning models (LeNet, UNet, DeepLabV3+). The pipeline includes exploratory data analysis (EDA), preprocessing, model training, and evaluation, with a final model based on UNet with a DenseNet121 backbone, using channels 1 and 4 at 512x512 resolution.

## Project Pipeline

The following diagram illustrates the project pipeline

![Project Pipeline](https://github.com/Menna-Ahmed7/Cloud-Segmentation/raw/main/pipeline.png)


1. **Literature Review**  
   - Investigated classical methods (Fmask, Random Forest) and deep learning models (LeNet, UNet, DeepLabV3+).
   - Selected Random Forest, LeNet, UNet (ResNet, DenseNet121 backbones), and DeepLabV3+ (ResNet, EfficientNet-B7 backbones) based on prior research.

2. **Data Exploration (EDA)**  
   - Analyzed 10,211 images (3,528 fully cloudy, 5,888 partially cloudy, 795 cloud-free).
   - Verified no missing masks, found 2 duplicate image-mask pairs.
   - Plotted mask pixel value distribution and band correlations (bands 1 and 4 least correlated, bands 1, 2, 3 more correlated with masks).
   - Identified mislabeled data through RGB visualization.

3. **Preprocessing**  
   - Selected bands 1 and 4 to reduce redundancy.
   - Manually filtered mislabeled images.
   - Resized images to 70% for DeepLabV3 to address GPU memory issues.

4. **Model Selection and Training**  
   - Final model: UNet with DenseNet121 backbone, channels 1 and 4, 512x512 resolution, confidence threshold of 0.7, trained for 30 epochs with a learning rate of 0.00025.
   - Results: Training Dice: 0.8994, Validation Dice: 0.9024, Test Dice: 0.8937.

5. **Model Evaluation**  
   - Conducted 13 trials testing LeNet, UNet (ResNet, DenseNet121), DeepLabV3+ (ResNet, EfficientNet-B7), Random Forest various thresholds, Dice Loss, and image sizes.
   - Best leaderboard score: UNet with DenseNet121, channels 1 and 4, 512x512 (0.7900 public, 0.7700 private).

6. **Enhancements and Future Work**  
   - Proposed improvements: semi-automated mislabeled data filtering, adaptive thresholding, targeted data augmentation, hybrid loss functions, model ensembling.
   - Future work: multi-resolution training, lightweight backbones (GANs), additional bands.

## Repository Structure

```
Cloud-Segmentation/
├── src/                                                                    # Project Implementation
│   ├── notebooks/                                                          # Notebooks folder
│         ├── eda_and_unet_train.ipynb                                      # EDA and UNet Model notebook 
│         ├── random_forest_segmentation.ipynb                              # Random forest notebook
|         ├── random-forest-cloud-segmentation-validation.ipynb             # Random forest notebook with validation trials
|         ├── random-forest-cloud-segmentation-validation-and-test.ipynb    # Random forest notebook with validation and test trials
│   ├── scripts/                                                            # Scripts folder
│         ├── profiler.py                                                   # Profiler script
│         ├── run_inference.py                                              # Inference script
├── README.md                                                               # This file
├── model.zip                                                               # The model file
├── team_12.csv                                                             # Output file (csv)
├── model_logs.txt                                                          # The model size and parameters

```

# How to Run

## profiler.py 
   ```bash
python profiler.py 1 2 512 512
   ```

## run_inference.py

1. Clone the repository:
   ```bash
   git clone https://github.com/Menna-Ahmed7/Cloud-Segmentation.git
   ```
2. Unzip the model file
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install segmentation-models-pytorch  
   pip install scikit-image
   pip install tifffile 
   ```

4. Add the required paths for the following variables:
- **model_path (after unzipping the model file)**
- **submission_sample_path**
- **test_folder**
   


## The notebooks

1. Import the notebook into kaggle:

2. Add the dataset as an input
  - Dataset link: https://www.kaggle.com/datasets/aliaagheis/cloud-masking-dataset


## Results

The final model (UNet with DenseNet121, channels 1 and 4, 512x512, threshold 0.7) achieved:
- Training: Accuracy 94.00%, Dice 0.8994
- Validation: Accuracy 94.14%, Dice 0.9024
- Test: Accuracy 93.44%, Dice 0.8937
- Leaderboard: Public 0.7900, Private 0.7700 (8th place)



## Contributors <a name = "Contributors"></a>

<table>
  <tr>
    <td align="center">
    <a href="https://github.com/Menna-Ahmed7" target="_black">
    <img src="https://avatars.githubusercontent.com/u/110634473?v=4" width="150px;" alt="https://github.com/Menna-Ahmed7"/>
    <br />
    <sub><b>Mennatallah Ahmed</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/MostafaBinHani" target="_black">
    <img src="https://avatars.githubusercontent.com/u/119853216?v=4" width="150px;" alt="https://github.com/MostafaBinHani"/>
    <br />
    <sub><b>Mostafa Hani</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/MohammadAlomar8" target="_black">
    <img src="https://avatars.githubusercontent.com/u/119791309?v=4" width="150px;" alt="https://github.com/MohammadAlomar8"/>
    <br />
    <sub><b>Mohammed Alomar</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/mou-code" target="_black">
    <img src="https://avatars.githubusercontent.com/u/123744354?v=4" width="150px;" alt="https://github.com/mou-code"/>
    <br />
    <sub><b>Moustafa Mohammed</b></sub></a>
    </td>
  </tr>
 </table>

## Acknowledgments

We thank our instructors and teaching assistants at the Computer Engineering Department for their guidance and support throughout the project.
