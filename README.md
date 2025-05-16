# Shadow Hand Motion Prediction

A DCT transformer-based system for predicting human hand movements for control and predictions for the Dexterous Shadow Hand.

## Overview

This project addresses the challenge of latency in teleoperation systems by implementing a predictive model for hand motion. By forecasting future hand positions 80-1000ms ahead, the system enables smoother control of robotic hands even with network delays.

## Key Features

- **Motion Data Conversion**: Transforms data from Ninapro and Rokoko datasets into Shadow Hand joint angles
- **DCT-Enhanced Transformer**: Implements a frequency-domain transformer model for accurate motion prediction
- **Progressive Decoder**: Uses anatomically-aware decoding that respects hand biomechanics
- **Multi-Horizon Prediction**: Supports prediction windows from 80ms to 1000ms with stable error progression
- **Visual Validation**: Integration with Shadow Hand simulator for testing and visualization

## Installation

```bash
# Clone the repository
git clone https://github.com/Kanarifuglen/ShadowHandThesis.git
cd ShadowHandThesis

# Install dependencies
pip install -r requirements.txt

**Note**: To use the Shadow Hand simulator, you need to install the Shadow Robot Company's software. See the [installation guide](https://github.com/sinlab-uio/rokoko-to-robots) for more details.
```

## Usage

### Dataset Conversion
```python
# Convert Ninapro dataset to Shadow Hand angles
python scripts/makeShadowDataset.py --subject.mat --output dataset/subject1.npz

# Process full dataset with multi-threading
python scripts/makeFullShadowSet.py --threads 4 --output dataset/full_dataset.h5
```

### Model Training
```python
# Train the model with default parameters
python train.py --dataset dataset/shadowDataset.h5 --epochs 10

# Evaluate on test set
python evaluate.py --model models/best_model.pt --dataset dataset/shadowDataset.h5
```

### Visualization
```python
# Visualize predictions in Shadow Hand simulator
python converter.py --sendAngles.py x.mat
```

## Project Structure

```
├── datasets/              # Dataset processing modules 
├── transformer/           # Transformer model implementation
├── conversions/           # Conversion and utility scripts
├── evaluation/            # Evaluation metrics and analysis
└── requirements.txt       # Project dependencies
```

## Citation

If you use this code in your research, please cite:

```
@mastersthesis{kanaris2025hand,
  title={Transformer-Based Hand Motion Prediction for Robotic Control},
  author={Kanaris, Jarle André Rivas},
  year={2025},
  school={University of Oslo}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

References:
[1] Mao, W., Liu, M., Salzmann, M., & Li, H. (2019). Learning Trajectory Dependencies for Human Motion Prediction. In Proceedings of the IEEE International Conference on Computer Vision (pp. 9489-9497). https://github.com/wei-mao-2019/LearnTrajDep

[2] Cai, Y., Huang, L., Wang, Y., Cham, T. J., Cai, J., Yuan, J., Liu, J., Yang, X., Zhu, Y., Shen, X., Liu, D., Liu, J., & Thalmann, N. M. (2020). Learning Progressive Joint Propagation for Human Motion Prediction. In Proceedings of the European Conference on Computer Vision (ECCV).

[3] Shi, Y., Srivastava, S., Kirchain, S., Gupta, S., Raue, F., Nguyen, J., Aydin, A., & Thomaz, A. L. (2022). Motion Transformer with Global-Local Motion Embedding. Neural Information Processing Systems (NeurIPS) Workshop on Motion, Action and Perception. https://github.com/eth-ait/motion-transformer
