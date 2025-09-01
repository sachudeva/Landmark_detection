
# Fetal Biometry Landmark Detection

This repository contains implementations of two deep learning approaches for detecting biparietal diameter (BPD) and occipitofrontal diameter (OFD) landmark points in fetal ultrasound images.

## Approaches

### 1. Direct Coordinate Regression (Baseline)
- **Architecture**: Modified ResNet18 with final fully connected layer outputting 8 coordinate values
- **Loss Function**: Mean Squared Error (MSE)
- **Method**: Direct regression of (x,y) coordinates for 4 landmark points
- **Key Features**: Standard data augmentation including flips, rotations, and color jitter

### 2. Heatmap Regression (Advanced)
- **Architecture**: Encoder-decoder based on ResNet18 with upsampling layers
- **Loss Function**: Mean Squared Error (MSE) on Gaussian heatmaps
- **Method**: Predicts heatmaps for each landmark, then extracts coordinates via argmax
- **Key Features**: Enhanced augmentation including affine transformations, Gaussian blur, and more aggressive color jitter

## Results

### Direct Coordinate Regression
```
Epoch 1/2, Train Loss: 0.003611, Val Loss: 0.016582
Epoch 2/2, Train Loss: 0.004370, Val Loss: 0.016978
```

### Heatmap Regression
```
Mean Absolute Error: 20.31 pixels
Success rate with 2px threshold: 12.60%
Success rate with 5px threshold: 32.70%
Success rate with 10px threshold: 52.40%
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fetal_biometry_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- torch>=1.9.0
- torchvision>=0.10.0
- numpy>=1.21.0
- pandas>=1.3.0
- matplotlib>=3.4.0
- opencv-python>=4.5.0
- scikit-learn>=0.24.0

## Usage

### Training

1. **Direct Coordinate Regression**:
```bash
python train_direct.py --csv_path data/landmarks_data.csv --image_dir data/images --epochs 50
```

2. **Heatmap Regression**:
```bash
python train_heatmap.py --csv_path data/landmarks_data.csv --image_dir data/images --epochs 50
```

### Evaluation

```bash
python evaluate.py --model weights/resnet18_heatmap_detector.pth --csv_path data/landmarks_data.csv --image_dir data/images
```

### Visualization

```bash
python visualize.py --model weights/resnet18_heatmap_detector.pth --csv_path data/landmarks_data.csv --image_dir data/images --output_dir results/
```

## Data Format

The CSV file should contain the following columns:
- `image_name`: Filename of the ultrasound image
- `ofd_1_x`, `ofd_1_y`: First OFD landmark coordinates
- `ofd_2_x`, `ofd_2_y`: Second OFD landmark coordinates  
- `bpd_1_x`, `bpd_1_y`: First BPD landmark coordinates
- `bpd_2_x`, `bpd_2_y`: Second BPD landmark coordinates

Example:
```
image_name,ofd_1_x,ofd_1_y,ofd_2_x,ofd_2_y,bpd_1_x,bpd_1_y,bpd_2_x,bpd_2_y
000_HC.png,361,12,339,530,481,16,664,318
```

## Model Architectures

### Direct Regression Model
- Backbone: ResNet18 (pretrained on ImageNet)
- Final layer: Fully connected layer with 8 outputs
- Input size: 256×256 pixels
- Output: 8 normalized coordinates [0, 1]

### Heatmap Regression Model  
- Encoder: ResNet18 (pretrained)
- Decoder: Upsampling layers with convolutional blocks
- Output: 8 heatmaps (64×64) with Gaussian distributions
- Heatmap decoding: Coordinate extraction via argmax

## Data Augmentation

### Direct Regression
- Random horizontal flip (p=0.5)
- Random rotation (±10 degrees)
- Color jitter (brightness=0.2, contrast=0.2)

### Heatmap Regression (Enhanced)
- Random horizontal flip (p=0.5)  
- Random rotation (±15 degrees)
- Color jitter (brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
- Random affine transformations (translation=0.1, scale=0.9-1.1)
- Gaussian blur (kernel=3, σ=0.1-2.0)

## Performance Metrics

- **Mean Absolute Error (MAE)**: Average pixel distance between predicted and true landmarks
- **Success Rate**: Percentage of predictions within specified pixel thresholds (2px, 5px, 10px)

## Key Findings

1. The heatmap approach provides better spatial awareness and robustness compared to direct regression
2. Enhanced augmentation significantly improves model generalization for ultrasound images
3. Heatmap regression achieves 52.4% success rate within 10px threshold
4. Direct regression converges faster but may have higher localization error

## Future Work

1. Experiment with different network architectures (U-Net, HRNet, Vision Transformers)
2. Implement additional loss functions (Dice loss, Focal loss)
3. Add attention mechanisms for better feature focusing
4. Explore multi-task learning with segmentation and landmark detection
5. Implement test-time augmentation for improved robustness

## References

1. Regressing Heatmaps for Multiple Landmark Localization using CNNs
2. Cephalometric Landmark Detection by Attentive Feature Pyramid Fusion and Regression-Voting  
3. U-Net: Convolutional Networks for Biomedical Image Segmentation
4. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation

## License

This project is for research purposes. Please contact the authors for licensing information.
careers-india@originhealth.ai
