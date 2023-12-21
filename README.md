<h1 align="center">SleepBoost: A Multi-level Tree-based Ensemble Model for Automatic Sleep Stage Classification</h1>
<!---  
## Paper Link [Read the Full Paper Here](URL_to_Paper) ## Authors List
- Author 1, Affiliation
- Author 2, Affiliation
- Author 3, Affiliation
- *Add or remove authors as needed*
--->

## Abstract
Clinicians' workload in sleep studies has lightened due to advances in Automatic Sleep Stage Classification (ASSC). While deep learning (DL) models, with their intricate architectures like attention mechanisms and transfer learning, have made significant strides, their "black-box" nature raises trust concerns among medical practitioners. Addressing this, we introduce SleepBoost, a transparent Multi-level Tree-based Ensemble Model tailored for ASSC. Utilizing our crafted Feature Engineering Block (FEB), we extracted 42 time and frequency domain features, of which 17 were selected via a mutual information score threshold (>0.23). SleepBoost's architecture integrates three fundamental linear models into a multi-level tree structure, enhanced with a reward-based adaptive weight allocation. On the Sleep-EDF-20 dataset, SleepBoost achieved an accuracy of 86.3%, an f1-score of 80.9%, and a Cohen kappa score of 0.807, surpassing leading DL models in ASSC. An ablation study further emphasizes the pivotal role of feature selection.

## Overview of the Method

<br> ![SleepBoost](/figures/SleepBoost.jpg)
<br> *General architecture of SleepBoost. We trained Random FOrest (RF), Light Gradient Boosting (LGBoost), and Categorical Boosting (CatBoost) as a unit block model for SleepBoost using the training dataset. Adaptive weight calculation is initiated using the prediction of the unit block models. Finally, a weighted score is calculated to predict the sleep stage*
<!---
## Feature Tables
*Describe the tables included in your README.*

### Table 1: Dataset Overview

| Feature | Description | Details |
|---------|-------------|---------|
| Feature 1 | Description 1 | Details 1 |
| Feature 2 | Description 2 | Details 2 |
| ... | ... | ... |

*Add more tables as needed.*
--->
## Extracted Features

#### Time Domain Features with Computational Equations
<br> ![](/figures/Table-03.png)

#### Direct Frequency Domain Feature Symbols with Computational Method
<br> ![](/figures/Table-04.png)

#### Derived Frequency Domain Feature Symbols with Computational Method
<br> ![](/figures/Table-05.png)



## Additional Figures

![Ablation Study 01](/figures/Ab2.jpg)
<br> *Comparison of Manual Sleep Stage Labelling with SleepBoost Model’s Prediction. The actual Label represents the result of manual staging by Experts, while the Predicted Label is the result of the SleepBoost 
Model*

![Ablation Study 02](/figures/Ab1.jpg)
<br> *Comparison of performance in different variants of SleepBoost. M1 (all features + balanced weight), M2 (selected features + balanced weight), M3 (all features + adaptive weight), and M4 (selected features + adaptive weight) represent four variants of SleepBoost with the combinations of features and weight allocation.*

![Confusion Matrices](/figures/CM.jpg)
<br> *Confusion Matrices of all the Models (a) Support Vector Machine (SVM) (b) Adaptive Boosting (AdaBoost) (c) Random Forest (RF) (d) Categorical Boosting (CatBoost) (e) Light Gradient Boosting (LGBoost) and (f) SleepBoost*


![ROC](/figures/ROC.jpg)
<br> *Comparison of Area under the Receiver Operating Curve (AUC-ROC) among the conventional models and SleepBoost. Left: using all extracted features Right: using selected features of FEB*
<!---
![Figure 2](/figures/figure2.png)
*Figure 2: Caption describing this figure.*

![Figure 2](/figures/figure2.png)
*Figure 2: Caption describing this figure.*

![Figure 2](/figures/figure2.png)
*Figure 2: Caption describing this figure.*

*Add more figures as needed.*
--->
## Description of the GitHub Repository

This repository contains all the necessary code, data, and instructions to replicate the findings of our paper. The repository is structured as follows:

- `src/`: Source code used in the research.
- `figures/`: Figures and graphs used in the paper.
<!--- - `data/`: Data files and preprocessing scripts. --->
<!--- - `docs/`: Further documentation on the code and the research. --->
<!--- - *Include any additional relevant directories and their descriptions.* --->

## Prepare dataset ##
We evaluated our SleepBoost with [Sleep-EDF](https://www.physionet.org/content/sleep-edfx/1.0.0/) dataset.

For the [Sleep-EDF]([https://physionet.org/pn4/sleep-edfx/](https://www.physionet.org/content/sleep-edfx/1.0.0/)) dataset, you can run the following scripts to download SC subjects.

    cd data
    chmod +x download_physionet.sh
    ./download_physionet.sh

Then run the following script to extract specified EEG channels and their corresponding sleep stages.

    python prepare_physionet.py --data_dir data --output_dir data/eeg_fpz_cz --select_ch 'EEG Fpz-Cz'
    python prepare_physionet.py --data_dir data --output_dir data/eeg_pz_oz --select_ch 'EEG Pz-Oz'

## Create a virtual environment with venv/conda

```bash
python3 -m venv sleepboost
source .sleepboost/bin/activate
python3 -m pip install -r requirements.txt
```


## Citation

If you use our code or methodology in your work, please cite our paper as follows:

<!---

```bibtex
@article{sleepboost2023,
  title={SleepBoost: A Multi-level Tree-based Ensemble Model for Automatic Sleep Stage Classification},
  author={Author 1 and Author 2 and Author 3},
  journal={Journal Name},
  volume={xx},
  number={xx},
  pages={xx--xx},
  year={2023},
  publisher={Publisher}
}
--->

