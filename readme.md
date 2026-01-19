...existing code...

# Deep Learning Models for Brain Tumour Classification

A Colab-ready project containing end-to-end pipelines (from data extraction to training and evaluation) for classifying MRI brain scans into 4 classes: `glioma`, `meningioma`, `notumor`, `pituitary`. The main notebook is [capstone.ipynb](capstone.ipynb).

Dataset:
- Original dataset used: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Quick links:
- Notebook: [capstone.ipynb](capstone.ipynb)
- This README: [README.md](README.md)

Repository layout
- [capstone.ipynb](capstone.ipynb) — main Colab notebook with:
  - ZIP extraction & folder tree: [`print_tree`](capstone.ipynb)
  - Simple CNN from scratch: [`make_model`](capstone.ipynb)
  - ResNet50 transfer-learning pipeline: [`make_resnet50_model`](capstone.ipynb)
  - EfficientNetB0 transfer-learning pipeline: [`make_efficientnet_model`](capstone.ipynb)
  - Dataset preparation: [`prepare`](capstone.ipynb)
  - On-the-fly augmentation: [`data_augmentation`](capstone.ipynb)

Highlights / What the notebook does
1. Extracts a ZIP archive (user-provided) into an `extracted` folder and prints the folder tree using [`print_tree`](capstone.ipynb).
2. Loads datasets with Keras `image_dataset_from_directory` from:
   - train: `/content/extracted/Training`
   - test: `/content/extracted/Testing`
3. Prepares tf.data pipelines with normalization, shuffling and prefetching (`AUTOTUNE`) — see [`prepare`](capstone.ipynb).
4. Offers three model choices:
   - Custom CNN: [`make_model`](capstone.ipynb)
   - ResNet50 transfer-learning: [`make_resnet50_model`](capstone.ipynb)
   - EfficientNetB0 transfer-learning: [`make_efficientnet_model`](capstone.ipynb)
5. Trains (including optional fine-tuning), evaluates on test set, prints classification reports and confusion matrices, and saves models:
   - ResNet saved to `/content/resnet50_finetuned_mri.h5`
   - EfficientNet saved to `/content/efficientnetb0_finetuned_mri.h5`

How to run (recommended: Google Colab)
1. Upload the project and dataset ZIP to Colab or mount Google Drive.
2. Open [capstone.ipynb](capstone.ipynb) in Colab.
3. Run the first cell and provide the path to the ZIP archive when prompted (or edit `zip_path`).
4. Run cells sequentially. The notebook will:
   - extract the archive into `extracted/`
   - create train/validation/test datasets
   - train the selected model(s)
   - evaluate and save model files

Key configuration variables (inside notebook)
- `train_dir`, `test_dir` — dataset directories (defaults in notebook: `/content/extracted/Training`, `/content/extracted/Testing`)
- `IMG_SIZE` — input image size (e.g. `(224, 224)`)
- `BATCH_SIZE`, `EPOCHS`, `SEED`
- `AUTOTUNE` — tf.data autotune for performance

Training tips
- Use GPU runtime in Colab (Runtime → Change runtime type → GPU).
- Reduce `BATCH_SIZE` if you hit OOM.
- For transfer learning:
  - Start with base models frozen (the notebook sets `base_trainable=False`).
  - Optionally unfreeze top layers and run a few epochs with a lower LR (see `fine_tune_at` and recompile steps in the ResNet/EfficientNet cells).
- Monitor validation accuracy/loss and use early stopping if needed.

Evaluation & outputs
- The notebook computes:
  - test loss & accuracy
  - classification report (precision, recall, f1) using `classification_report`
  - confusion matrix (counts and normalized)
- Example printed output in the notebook: "Classes found: ['glioma', 'meningioma', 'notumor', 'pituitary']".

Files produced by the notebook
- `/content/resnet50_finetuned_mri.h5` — ResNet-based model (if section executed)
- `/content/efficientnetb0_finetuned_mri.h5` — EfficientNet-based model (if section executed)

Where to look in the notebook
- ZIP extraction & printing tree: see the first cell with [`print_tree`](capstone.ipynb)
- Basic CNN pipeline & training: search for [`make_model`](capstone.ipynb)
- ResNet50 pipeline & fine-tuning: see [`make_resnet50_model`](capstone.ipynb)
- EfficientNetB0 pipeline: see [`make_efficientnet_model`](capstone.ipynb)
- Data preparation utilities: [`prepare`](capstone.ipynb) and [`data_augmentation`](capstone.ipynb)

Reproducibility
- Fix `SEED` for deterministic shuffling in dataset splits.
- Save model weights and training history if you want to resume or analyze runs offline.

Notes
- The notebook assumes the dataset follows the folder structure:
  - extracted/Training/<class_name>/*.jpg
  - extracted/Testing/<class_name>/*.jpg
  - Classes used in examples: `glioma`, `meningioma`, `notumor`, `pituitary`
- Designed for Colab (GPU available) but runnable locally with appropriate GPU/TF setup.

License / Attribution
- Dataset source: Kaggle dataset linked above.
- This repository is an educational pipeline; adjust preprocessing, augmentation, and model hyperparameters for production research.

If you want, I can:
- Generate a condensed "quickstart" cell to paste at the top of [capstone.ipynb](capstone.ipynb).
- Add an example Colab shortcut link (requires hosting the notebook in a public repo).
