# CIFAR10 Image Classification
- Conv2d(128, 10, k=1)
- Log Softmax

### Key Features
- No MaxPooling (replaced with strided convolutions)
- Uses Depthwise Separable Convolution in C2
- Uses Dilated Convolution in C3
- Total Parameters: ~198k
- Final Receptive Field: 45x45

## Image Augmentation Techniques

Using Albumentations library with the following transforms:

1. **HorizontalFlip**
    - Probability: 0.5

2. **ShiftScaleRotate**
    - Shift Limit: 0.1
    - Scale Limit: 0.1
    - Rotate Limit: 15°
    - Probability: 0.5

3. **CoarseDropout**
    - Max Holes: 1
    - Max Height: 16px
    - Max Width: 16px
    - Min Holes: 1
    - Min Height: 16px
    - Min Width: 16px
    - Fill Value: Dataset Mean
    - Probability: 0.5

## Sample Images

### Before Augmentation
![Before Augmentation](sample_images/cifar10_grid_before_augmentation.png)

### After Augmentation
![After Augmentation](sample_images/cifar10_grid_after_augmentation.png)

## Training Summary 
| Metric           | Value     |
|------------------|-----------|
| Total Parameters  | 198.2k   |
| Best Test Acc    | 85.23%   |
| Final Train Acc   | 84.89%   |
| Final Test Acc    | 84.92%   |
| Learning Rate     | 0.01     |
| Momentum          | 0.9      |
| Batch Size        | 128      |
| Total Epochs      | 15       |

## Epoch-wise Performance
|   Epoch | Train Accuracy | Test Accuracy |
|---------|----------------|----------------|
|       1 | 10.94%        | 46.94%         |
|       2 | 13.28%        | 53.69%         |
|       3 | 12.24%        | 60.35%         |
|       4 | 13.09%        | 62.43%         |
|       5 | 13.44%        | 64.00%         |
|       6 | 13.67%        | 66.21%         |
|       7 | 14.73%        | 68.00%         |
|       8 | 15.33%        | 67.73%         |
|       9 | 16.15%        | 71.76%         |
|      10 | 16.64%        | 72.31%         |
|      11 | 16.62%        | 73.75%         |
|      12 | 16.67%        | 73.99%         |
|      13 | 16.77%        | 75.33%         |
|      14 | 17.08%        | 76.57%         |


Epoch-wise Performance
| Epoch | Train Accuracy | Test Accuracy |
|---------|----------------|---------------|
| 1 | 10.94% | 46.94% |
| 2 | 13.28% | 53.69% |
| 3 | 12.24% | 60.35% |
| 4 | 13.09% | 62.43% |
| 5 | 13.44% | 64.00% |
| 10 | 16.64% | 72.31% |
| 15 | 17.76% | 76.56% |
| 20 | 18.67% | 77.51% |
| 25 | 19.56% | 79.15% |
| 30 | 20.83% | 78.73% |
| 35 | 21.63% | 80.18% |
| 40 | 22.36% | 80.68% |
| 45 | 22.90% | 82.27% |
| 50 | 23.38% | 82.82% |
| 55 | 23.88% | 83.01% |
| 60 | 24.30% | 83.80% |
| 65 | 24.76% | 83.50% |
| 70 | 25.20% | 83.35% |
| 75 | 25.52% | 83.79% |
| 80 | 25.85% | 84.11% |
| 85 | 26.31% | 83.81% |
| 90 | 26.61% | 84.43% |
| 95 | 27.11% | 84.90% |
| 100 | 27.31% | 85.38% |
