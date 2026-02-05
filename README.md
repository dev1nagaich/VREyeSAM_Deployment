---
title: VREyeSAM - Iris Segmentation
emoji: 👁️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# VREyeSAM: Virtual Reality Non-Frontal Iris Segmentation

![VREyeSAM Demo](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)

## 🎯 Overview

VREyeSAM is a robust iris segmentation framework designed specifically for non-frontal iris images captured in virtual reality and head-mounted device environments. Built on Meta's Segment Anything Model 2 (SAM2) with a novel uncertainty-weighted loss function, VREyeSAM achieves state-of-the-art performance on challenging VR/AR iris segmentation tasks.

## 🚀 Features

- **Upload & Segment**: Upload any non-frontal iris image for instant segmentation
- **Binary Mask Generation**: Get precise binary segmentation masks
- **Iris Extraction**: Automatically extract and display the iris region as a rectangular strip
- **Visualization Options**: View overlay masks and probabilistic confidence maps
- **Download Results**: Save all segmentation outputs with one click

## 📊 Performance Metrics

- **Precision**: 0.751
- **Recall**: 0.870
- **F1-Score**: 0.806
- **Mean IoU**: 0.647

Evaluated on the VRBiom dataset, VREyeSAM significantly outperforms existing segmentation methods.

## 🔬 Technical Details

### Architecture
VREyeSAM leverages:
- **Base Model**: SAM2 (Segment Anything Model 2) with Hiera-Small backbone
- **Fine-tuning**: Custom uncertainty-weighted hybrid loss function
- **Training Data**: VRBiomSegM dataset with non-frontal iris images
- **Inference**: Point-prompt based segmentation with ensemble predictions

### Key Innovations
1. **Quality-aware Pre-processing**: Automatically filters partially/fully closed eyes
2. **Uncertainty-weighted Loss**: Adaptively balances multiple learning objectives
3. **Multi-point Sampling**: Uses 30 random points for robust predictions
4. **Probabilistic Masking**: Generates confidence-weighted segmentation

## 🎓 Citation

If you use VREyeSAM in your research, please cite:

```bibtex
@article{sharma2025vreyesam,
  title={VREyeSAM: Virtual Reality Non-Frontal Iris Segmentation using Foundational Model with Uncertainty Weighted Loss},
  author={Sharma, Geetanjali and Nagaich, Dev and Jaswal, Gaurav and Nigam, Aditya and Ramachandra, Raghavendra},
  conference={IJCB},
  year={2025}
}
```

## 👥 Authors

- **Geetanjali Sharma** - Indian Institute of Technology Mandi, India
- **Dev Nagaich** - Indian Institute of Technology Mandi, India
- **Gaurav Jaswal** - Division of Digital Forensics, Directorate of Forensic Services, Shimla, India
- **Aditya Nigam** - Indian Institute of Technology Mandi, India
- **Raghavendra Ramachandra** - Norwegian University of Science and Technology (NTNU), Norway

## 📧 Contact

For dataset access or questions:
- **Email**: geetanjalisharma546@gmail.com
- **GitHub**: [VREyeSAM Repository](https://github.com/GeetanjaliGTZ/VREyeSAM)

## 🔗 Resources

- [Paper on ResearchGate](https://www.researchgate.net/publication/400248367_VREyeSAM_Virtual_Reality_Non-Frontal_Iris_Segmentation_using_Foundational_Model_with_uncertainty_weighted_loss)
- [GitHub Repository](https://github.com/GeetanjaliGTZ/VREyeSAM)
- [Model Weights on Hugging Face](https://huggingface.co/devnagaich/VREyeSAM)

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Meta AI for the Segment Anything Model 2 (SAM2)
- VRBiom dataset contributors
- Indian Institute of Technology Mandi
- Norwegian University of Science and Technology

## 🛠️ Usage Instructions

1. **Upload Image**: Click on the upload button and select a non-frontal iris image
2. **Segment**: Click the "Segment Iris" button to process the image
3. **View Results**: Explore the binary mask, overlay, and extracted iris strip
4. **Download**: Save any of the results using the download buttons

## ⚙️ Model Details

- **Model Type**: Image Segmentation
- **Base Architecture**: SAM2 (Hiera-Small)
- **Training Dataset**: VRBiomSegM (contact for access)
- **Input Size**: Up to 1024px (auto-resized)
- **Output**: Binary mask + Probabilistic confidence map
- **Device**: CUDA GPU (falls back to CPU if unavailable)

## 🔍 Use Cases

- **Biometric Authentication**: Secure iris recognition in VR/AR environments
- **Medical Applications**: Iris analysis in non-ideal capture conditions
- **Research**: Benchmark for non-frontal iris segmentation
- **VR/AR Development**: Integration into head-mounted devices

---

**Note**: This is a research prototype. For production use, please contact the authors.