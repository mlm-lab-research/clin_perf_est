# Clinical Performance Estimation

Code to reproduce results for the paper **“Label‑free estimation of clinically relevant performance metrics under distribution shifts”**

---

## Setup Environment

1. **Clone this repo**

   ```bash
   git clone <REPO_URL>
   cd <REPO_DIR>
   ```
2. **Create and activate Python environment**

   ```bash
   conda env create -f environment.yml
   conda activate clinical-perf


---

## Directory Structure

```
repo/
├── config/                
│   ├── plot_style.txt
├── figures/                       # folder for generated figures
├── notebooks/              
│   ├── controlled_shifts.ipynb    # Notebook to reproduce the figures for controlled prevalence and covariate shifts
│   ├── WILD_shifts.ipynb          # Notebook to reproduce the figures for shifts in the WILD
├── outputs/                       # Stores the calculated metrics for covshift and prevalence shift
├── results/                       
│   ├── artifact_array.npy         # Array keeping track of which test set indices are with the artifact
│   ├── CBPE_metrics_dict.pkl      # File containing the CBPE estimated metrics.
│   └── ...
└── utils/
│    ├── __init__.py
│    ├── metrics.py
│    ├── performance_estimation_methods.py
│    ├── plots.py
│    └── utils.py
└── environment.yml       # conda env specification
```


---
## Data disclaimer
Model training and metrics calculation in the WILDS were performed separately, and the results are stored in the *results/* subfolder for each estimation technique.
For the controlled shifts, the softmax outputs and labels from the classification model on Pleural Effusion on cheXpertPlus data, as well as the softmax outputs of the biased classifier, are also stored in the *results/* subfolder.


---

## How to Run

### 1. Reproduce all paper figures
Follow the notebooks.

