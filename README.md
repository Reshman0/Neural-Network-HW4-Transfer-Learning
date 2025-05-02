# Transfer Learning Experiments: DenseNet201 & ResNet50

This repository contains Jupyter notebooks and helper scripts for transfer‐learning experiments on the UCMerced Land-use and Intel Image Classification datasets. We compare three augmentation schemes (baseline, AugA, AugB) across varying training set sizes (n = 20…1000) using two pretrained backbones (DenseNet201, ResNet50).

## Repository Structure

transfer-learning-project/
├── notebooks/
│   ├── 01\_data\_prep.ipynb      # Download & split raw data
│   ├── 02\_train\_ucmerced.ipynb # Experiments on UCMerced
│   ├── 03\_train\_intel.ipynb    # Experiments on Intel dataset
│   └── 04\_plots\_and\_report.ipynb
├── src/
│   ├── data\_utils.py           # split\_dataset, make\_loader
│   └── model\_utils.py          # get\_model, train/evaluate functions
├── requirements.txt            # Python dependencies
└── README.md                   # This file

> **Note:** The `data/` directory is **not** included in version control (it’s large). You must download and place the raw datasets yourself.

---

## 1. Clone & Install

```bash
git clone https://github.com/<your-username>/transfer-learning-project.git
cd transfer-learning-project
python3 -m venv .venv
source .venv/bin/activate    # Windows PowerShell: .\.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2. Download Datasets

### UCMerced Land-use

* **Kaggle:** [https://www.kaggle.com/datasets/abdulhasibuddin/uc-merced-land-use-dataset](https://www.kaggle.com/datasets/abdulhasibuddin/uc-merced-land-use-dataset)
* Unzip and place under `data/UCMerced/images/` with 21 class subfolders:

  ```
  data/
    UCMerced/
      images/
        agricultural/
        airplane/
        … (21 total)
  ```

### Intel Image Classification

* **Kaggle:** [https://www.kaggle.com/datasets/puneet6060/intel-image-classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
* Unzip and place under `data/Intel/`:

  ```
  data/
    Intel/
      seg_train/seg_train/<class>/
      seg_test/seg_test/<class>/
  ```

---

## 3. Prepare & Split Data

Open `notebooks/01_data_prep.ipynb` and **Run All**.
This will:

1. Read raw files in `data/UCMerced/images` and `data/Intel/seg_*`
2. Create split folders under `data/UCM_split/` and `data/Intel_split/`

---

## 4. Run Experiments

1. **UCMerced**

   * `notebooks/02_train_ucmerced.ipynb` trains DenseNet201 & ResNet50 (n = 20…1000, 3 augmentations, 5 epochs).

2. **Intel**

   * `notebooks/03_train_intel.ipynb` runs the same pipeline on the Intel dataset.

---

## 5. Generate Plots & Report

* Open `notebooks/04_plots_and_report.ipynb`.
* It produces 4 accuracy‐curve figures (train/val/test for each model-dataset combo) and guidance for your final report.

---

## Tips & Tricks

* **Speed up** by:

  * Reducing `num_epochs` or the `ns` list
  * Using `num_workers > 0` and `pin_memory=True` in DataLoader
* **Ensure GPU**:

  ```python
  import torch
  print(torch.cuda.is_available())
  ```
* **Prevent sleep** during long runs.

---

Happy experimenting! 🚀
