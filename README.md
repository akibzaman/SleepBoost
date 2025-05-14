# SleepBoost: A Multi-level Tree-based Ensemble Model for Automatic Sleep Stage Classification

This is the codebase for the paper [SleepBoost](https://link.springer.com/article/10.1007/s11517-024-03096-x).

<p align="center">
  <img src="./figures/GA.png" alt="SleepBoost Logo"/>
</p>

Neurodegenerative diseases often exhibit a strong link with sleep disruption, highlighting the importance of effective sleep stage monitoring. In this light, Automatic Sleep Stage Classification (ASSC) plays a pivotal role, now more streamlined than ever due to advances in deep learning (DL). However, the opaque nature of DL models can be a barrier to clinical adoption, due to trust concerns among medical practitioners. To bridge this gap, we introduce **SleepBoost**, a transparent multi-level tree-based ensemble model specifically designed for ASSC. Our approach includes:

* **Feature Engineering Block (FEB)**
  Extracts 42 time- and frequency-domain features; selects 17 with mutual information > 0.23.
* **Multi-level Ensemble**
  Trains Random Forest, LightGBM, and CatBoost as base learners in a tree structure.
* **Adaptive Weight Allocation**
  Novel reward-based mechanism to combine model outputs.

Tested on the Sleep-EDF-20 dataset, SleepBoost achieves:

* **Accuracy:** 86.3%
* **F1-score:** 80.9%
* **Cohen’s κ:** 0.807

These results outperform leading DL models in ASSC. An ablation study underscores the critical role of our selective feature extraction in enhancing both accuracy and interpretability—essential for clinical use.

---

## 📊 Overview of the Method

<p align="center">
  <img src="./figures/SleepBoost.jpg" alt="SleepBoost Architecture"/>
</p>

---

## 📁 Repository Structure

```text
.
├── data/
│   ├── download_physionet.sh    # Script to download Sleep-EDF data
│   └── prepare_physionet.py     # Extract specific EEG channels & labels
├── figures/                     # All paper figures
├── src/
│   ├── FeatureExtraction.py     # Extract time/frequency features
│   ├── FeatureSelection.py      # Rank & select features via MI
│   ├── SleepBoost.py            # Standalone model evaluation
│   ├── SleepBoostKFold.py       # 10-fold cross-validation
│   └── Metric.py                # Generate confusion matrices & ROC curves
├── supplementary/               # Supplementary materials
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🛠️ Preparing the Dataset

We use the [Sleep-EDF-20 dataset](https://www.physionet.org/content/sleep-edfx/1.0.0/).

```bash
cd data
chmod +x download_physionet.sh
./download_physionet.sh
```

Extract the EEG channels “EEG Fpz-Cz” and “EEG Pz-Oz”:

```bash
python prepare_physionet.py \
  --data_dir data \
  --output_dir data/eeg_fpz_cz \
  --select_ch "EEG Fpz-Cz"

python prepare_physionet.py \
  --data_dir data \
  --output_dir data/eeg_pz_oz \
  --select_ch "EEG Pz-Oz"
```

---

## ⚙️ Environment Setup

### Using `venv`

```bash
git clone <repo_url>
cd SleepBoost
python3.11 -m venv sleepboost
source sleepboost/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Using `conda`

```bash
git clone <repo_url>
cd SleepBoost
conda create -n sleepboost python=3.11
conda activate sleepboost
pip install -r requirements.txt
```

---

## ▶️ Running the Code

1. **Feature Extraction**

   ```bash
   python src/FeatureExtraction.py
   ```

2. **Feature Selection**

   ```bash
   python src/FeatureSelection.py 
   ```

3. **Metrics Generation (confusion matrices and ROC curves)**

   ```bash
   # Edit the datapath on line 68 of src/Metrics.py 
   python src/Metrics.py
   ```

4. **Standalone Test**

   ```bash
   # Edit the datapath on line 44 of src/SleepBoost.py
   python src/SleepBoost.py
   ```

5. **10-Fold Cross-Validation**

   ```bash
   # Edit the datapath on line 46 of src/SleepBoostKFold.py
   python src/SleepBoostKFold.py
   ```

---

## 📖 Citation

If you use our code or methodology, please cite:

```bibtex
@article{zaman2024sleepboost,
  title        = {SleepBoost: a multi-level tree-based ensemble model for automatic sleep stage classification},
  author       = {Zaman, Akib and Kumar, Shiu and Shatabda, Swakkhar and Dehzangi, Iman and Sharma, Alok},
  journal      = {Medical \& Biological Engineering \& Computing},
  volume       = {62},
  number       = {9},
  pages        = {2769--2783},
  year         = {2024},
  publisher    = {Springer}
}
```
