# Data

No data is stored in this repository. Download the datasets yourself (the notebooks used Kaggle copies) and check each dataset's license and terms before use.

## Expected layout

The original notebooks read class folders named `0`, `1`, `2`, ... in label order. `scripts/train.py` expects the same: arrange (or rename) each dataset so the class folders are numbered in the order below.

### `cassava` (5 classes)

```
cassava-disease/train/
├── 0/   cbb      (cassava bacterial blight)
├── 1/   cbsd     (cassava brown streak disease)
├── 2/   cgm      (cassava green mottle)
├── 3/   cmd      (cassava mosaic disease)
└── 4/   healthy
```

### `rice` (3 classes)

```
rice_leaf_diseases/
├── 0/   bacterial_leaf_blight
├── 1/   brown_spot
└── 2/   leaf_smut
```

### `maize` (4 classes, extra experiment not in the paper)

```
Maize Leaf Disease Dataset/
├── 0/   blight
├── 1/   common_rust
├── 2/   gray_leaf_spot
└── 3/   healthy
```

The Cassava and Maize class names are the ones printed in the notebooks' classification reports. The Rice names are the paper's three categories, in report order. Check them against your copy.

## Dataset sizes (from the paper)

- **Cassava:** 5,894 images in 5 classes. The notebook's 10% test split has 590 images, which matches.
- **Rice leaf:** **120 images** in 3 classes. The paper applies augmentation (horizontal flip, vertical flip, rotation) to the *training* set. The notebook's test split has 48 images, which is 10% of 480 = 4 x 120, so the notebook's Rice folder probably already held augmented copies. See the leakage caution in the README. If you rebuild this experiment, split the 120 original images first and augment only the training portion.
- **Maize:** not in the paper; 421 test images in the notebook.

## Preprocessing (as in the notebooks)

- Read with OpenCV (BGR channel order), resized to 128 x 128, scaled to [0, 1].
- Shuffled with seed 2, then split 90% / 10% (train / test) at random, without stratification.

## Sources

The notebooks used copies of these datasets from Kaggle; the notebooks themselves record only the folder names, so confirm the exact Kaggle dataset pages and their licenses for the copies you use.

- **Cassava:** the paper cites the cassava dataset as Oyewola, D. O., Dada, E. G., Misra, S., Damaševičius, R. *Detecting cassava mosaic disease using a deep residual convolutional neural network with distinct block processing.* PeerJ Computer Science 7:e352 (2021).
- **Rice:** the paper describes a 120-image, three-class rice leaf dataset (bacterial leaf blight, brown spot, leaf smut), cited there as its reference [36].
- **Maize:** a four-class maize leaf disease dataset (blight, common rust, gray leaf spot, healthy). Not described in the paper.

Cite the dataset publications when you publish results, and replace the generic entries above with the exact Kaggle pages once you have confirmed them.
