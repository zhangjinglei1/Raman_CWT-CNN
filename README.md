CWT-SerumCODE: Deep Learning-Enhanced Raman Spectroscopy for Lung Cancer Diagnosis
This repository contains the official implementation of CWT-SerumCODE (Continuous Wavelet Transform Serum Consensus Oncology Detection), a framework that combines spontaneous serum Raman spectroscopy with interpretable deep learning for rapid lung cancer screening.
📌 Project Overview
This study addresses the challenges of low signal-to-noise ratio (SNR) and complex biological backgrounds in spontaneous Raman spectra by:Transforming 1D Raman signals into 2D scalograms using Continuous Wavelet Transform (CWT).Utilizing a modified GoogLeNet (Inception v1) architecture for multi-scale feature extraction.Providing mechanistic transparency through Grad-CAM interpretability analysis.
🛠 Prerequisites
Python: 3.8+PyTorch: 1.10+Torchvision: 0.11+Other Dependencies: numpy, pandas, scikit-learn, matplotlib, PIL, PyWavelets (for CWT)
🚀 Model Architecture & Training
The core of our framework is a GoogLeNet backbone modified for binary classification:Input Size: 224 × 224 × 3Output: 2 nodes (Cancer vs. Control)Optimizer: Adam (beta_1=0.9, beta_2=0.99)Learning Rate: Initial 1e-4 with CosineAnnealingLR scheduling (T_max=200)Loss Function: Focal Loss (to handle potential class imbalance)Regularization: Dropout (0.5)
📊 Key ResultsValidation Accuracy: 90.5% (Independent cohort, $n=21$).Statistical Robustness: 1,000-iteration Bootstrap analysis and 5-fold cross-validation.AUC 95% CI: 0.828 – 0.975.
📝 UsagePreprocessing: Convert your .txt or .csv Raman data into 2D CWT scalograms using the gaus2 mother wavelet.Training: Run the training cells in CNN_lung_serum-pro.ipynb or execute the training script:Inference: Evaluate the model performance on the independent test set using the saved .pth weights.
⚠️ StatementThis research is a preliminary, proof-of-concept investigation. The current implementation is intended for scientific research purposes and is not yet approved for clinical diagnostic use.
📬 ContactFor any questions regarding the code or the paper, please open an issue in this repository.
