# SSVEP Decoding from EEG Signals: A Machine Learning Approach

## Overview

This repository contains a Jupyter Notebook (`ssvep_decoding_notebook.ipynb`) that demonstrates a comprehensive pipeline for decoding **Steady-State Visual Evoked Potentials (SSVEP)** from Electroencephalography (EEG) signals. SSVEPs are brain responses to rhythmic visual stimuli, which are commonly utilized in Brain-Computer Interface (BCI) applications to infer a user's focus of attention.

The notebook uses a publicly available SSVEP dataset from the **EEG-Notebooks** project and applies various traditional machine learning techniques, leveraging powerful Python libraries such as `MNE-Python`, `Pyriemann`, and `Scikit-learn`, to classify the target stimulus frequency (specifically, distinguishing between 20 Hz and 30 Hz stimuli).

This notebook is derived from [EEG-Notebooks Project Homepage](https://neurotechx.github.io/eeg-notebooks_v0.2/auto_examples/visual_ssvep/02r__ssvep_decoding.html#setup)

## Project Goals

* To showcase a complete pipeline for SSVEP decoding, from raw data loading to classification and result visualization.
* To compare the performance of different machine learning algorithms and feature extraction techniques in an SSVEP BCI context.
* To provide a clear, reproducible, and well-documented example for researchers, students, and practitioners in neurotechnology and machine learning.

## Features

* **Data Loading:** Utilizes `eegnb` utilities to fetch and load SSVEP EEG data.
* **Preprocessing:** Includes steps for event detection and epoching EEG signals.
* **Filter Bank Approach:** Implements a common technique to enhance frequency-specific SSVEP features by band-pass filtering and concatenating channels.
* **Feature Extraction:** Employs covariance matrices and Common Spatial Patterns (CSP).
* **Riemannian Geometry:** Leverages `Pyriemann` for Riemannian geometry-based feature transformations (Tangent Space) and classification (MDM).
* **Machine Learning Models:** Compares Logistic Regression and Linear Discriminant Analysis (LDA).
* **Robust Evaluation:** Uses Stratified Shuffle Split Cross-Validation to assess model performance reliably using AUC (Area Under the Curve) scores.
* **Results Visualization:** Presents a clear bar plot comparing the performance of different decoding pipelines.

## Getting Started

This notebook is designed to be easily runnable on cloud platforms like **Kaggle Notebooks** or **Google Colab**.

### Prerequisites

Ensure you have a working Python 3 environment. The necessary libraries will be installed within the notebook itself.

### How to Run

1.  **Open the Notebook:**
    * **On Kaggle:** Upload `ssvep_decoding_notebook.ipynb` to a new Kaggle Notebook or create a new notebook and paste the code.
    * **On Google Colab:** Upload `ssvep_decoding_notebook.ipynb` to Colab or create a new Colab notebook and paste the code.

2.  **Install Dependencies:**
    * The first code cells in the notebook contain `!pip install` commands. **It is crucial to run these cells first.** These commands will install `eegnb`, `mne`, `pyriemann`, `pylsl`, and `brainflow`.
    * Additionally, because `pylsl` relies on a system-level binary (`liblsl`), a separate `!apt-get install -y liblsl-dev` command is included.

3.  **Restart Kernel/Runtime (CRITICAL STEP):**
    * **After running the installation cells (especially the `apt-get` command), you MUST restart the notebook's Kernel/Runtime.** This allows the environment to recognize the newly installed system libraries.
        * **Kaggle:** Go to `File` -> `Restart Session` or `Run` -> `Restart Kernel`.
        * **Google Colab:** Go to `Runtime` -> `Restart Runtime`.

4.  **Run All Cells:**
    * After restarting, click "Run All" (or execute cells sequentially) from the beginning. All data loading, preprocessing, model training, and evaluation will be performed.

## Data Source

The dataset used in this notebook is sourced from the `visual-SSVEP` experiment within the **EEG-Notebooks** project. This dataset is designed for rapid experimentation and validation of BCI algorithms using consumer-grade EEG hardware. The `fetch_dataset` utility ensures the data is downloaded automatically if not present in your environment.

## Methodology

The core methodology follows these steps:

1.  **Raw Data Loading:** EEG recordings are loaded for a specific subject and session.
2.  **Event Detection & Epoching:** Stimulus onset events are identified, and the continuous EEG data is segmented into fixed-duration epochs around these events.
3.  **Filter Bank Preprocessing:** For each original EEG channel, multiple copies are created, each band-pass filtered around a specific target frequency (e.g., 20 Hz and 30 Hz). These filtered channels are then concatenated to form an augmented feature set.
4.  **Machine Learning Pipelines:**
    * **Covariance Matrix Estimation:** Covariance matrices are computed from the epoched, filtered data as a robust feature representation.
    * **Spatial Filtering (CSP):** Common Spatial Patterns are applied to maximize variance between the two stimulus classes.
    * **Riemannian Geometry:** Tangent Space mapping transforms the covariance matrices into a Euclidean space suitable for standard linear classifiers. Minimum Distance to Mean (MDM) is used as a direct Riemannian classifier.
    * **Classifiers:** Linear Discriminant Analysis (LDA) and Logistic Regression are used for the final classification step.
5.  **Cross-Validation:** Stratified Shuffle Split is employed to provide reliable performance estimates by repeatedly splitting the data into training and testing sets while preserving class proportions.
6.  **Performance Evaluation:** The Area Under the ROC Curve (AUC) is used as the primary metric to compare the decoding accuracy of different pipelines.

## Key Findings (Example based on typical results)

* The implementation effectively decodes SSVEP responses, achieving AUC scores well above chance level (0.5).
* Typically, pipelines incorporating **Common Spatial Patterns (CSP)** and/or **Riemannian Geometry** (e.g., Tangent Space mapping, MDM) tend to yield robust performance for SSVEP decoding, often outperforming simpler methods.
* The results highlight the effectiveness of integrating signal processing techniques with machine learning for neurophysiological data analysis.

## Dependencies

The primary Python libraries required are:

* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `mne`
* `scikit-learn`
* `pyriemann`
* `eegnb` (installed from GitHub)
* `pylsl`
* `brainflow`
