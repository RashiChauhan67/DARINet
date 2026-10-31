# Recorded results

Everything on this page is copied from the saved outputs of the original notebooks in [`../notebooks/`](../notebooks/). **None of it was re-run** for this repository.

Protocol of the recorded runs: random 90% / 10% split (seed 2). The 10% split was also used as `validation_data`, with early stopping and best-checkpoint saving on **validation loss**. Early stopping patience was 10 (Cassava) and 20 (Rice, Maize).

For Cassava the training cell is commented out in the notebook: the model was loaded from a saved checkpoint, and the training curves are from an earlier run.

All three models have the same architecture; only the output layer differs: 648,390 parameters (Cassava, 5 classes), 648,132 (Rice, 3 classes), 648,261 (Maize, 4 classes). The Cassava notebook prints a hand-estimated 9.30 GFLOPs (approximate) and a model size of 2.47 MB.

## Cassava (590 test images)

| Class | Precision | Recall | F1 | Support | AUC |
|---|---|---|---|---|---|
| cbb | 0.43 | 0.51 | 0.47 | 49 | 0.8555 |
| cbsd | 0.82 | 0.75 | 0.78 | 166 | 0.9263 |
| cgm | 0.78 | 0.61 | 0.68 | 76 | 0.9072 |
| cmd | 0.80 | 0.90 | 0.85 | 250 | 0.9403 |
| healthy | 0.85 | 0.71 | 0.78 | 49 | 0.9755 |
| accuracy | | | 0.77 | 590 | |
| macro avg | 0.74 | 0.70 | 0.71 | 590 | |
| weighted avg | 0.78 | 0.77 | 0.77 | 590 | |

## Rice (48 test images)

Test loss 0.0358, test accuracy 0.9792.

| Class | Precision | Recall | F1 | Support | AUC |
|---|---|---|---|---|---|
| bacterial_leaf_blight | 1.00 | 1.00 | 1.00 | 21 | 1.0000 |
| brown_spot | 1.00 | 0.94 | 0.97 | 18 | 0.9981 |
| leaf_smut | 0.90 | 1.00 | 0.95 | 9 | 0.9972 |
| accuracy | | | 0.98 | 48 | |
| macro avg | 0.97 | 0.98 | 0.97 | 48 | |
| weighted avg | 0.98 | 0.98 | 0.98 | 48 | |

With 48 images, one prediction is worth 2.08 percentage points of accuracy.

**Possible leakage.** The Rice dataset has 120 original images (flip and rotation augmentation is commonly applied to its training portion). The notebook's test split has 48 images, which is 10% of 480 = 4 x 120. That suggests the folder held augmented copies before the random split, so copies of the same leaf may appear in both training and test data. This could not be verified from the notebook.

## Maize (421 test images; extra experiment)

| Class | Precision | Recall | F1 | Support | AUC |
|---|---|---|---|---|---|
| blight | 0.89 | 0.96 | 0.92 | 121 | 0.9909 |
| common_rust | 0.96 | 0.98 | 0.97 | 123 | 0.9986 |
| gray_leaf_spot | 0.91 | 0.70 | 0.79 | 56 | 0.9844 |
| healthy | 0.98 | 1.00 | 0.99 | 121 | 1.0000 |
| accuracy | | | 0.94 | 421 | |
| macro avg | 0.94 | 0.91 | 0.92 | 421 | |
| weighted avg | 0.94 | 0.94 | 0.94 | 421 | |

AUC values are read from the legends of the ROC plots in the notebooks.
