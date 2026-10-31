# Data

No data is stored in this repository. Download the datasets yourself and check each dataset's license and terms before use.

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

### `maize` (4 classes, extra experiment)

```
Maize Leaf Disease Dataset/
├── 0/   blight
├── 1/   common_rust
├── 2/   gray_leaf_spot
└── 3/   healthy
```

The class names above are the ones printed in the notebooks' classification reports (Cassava and Maize) or inferred from the report order (Rice). Check them against your copy of each dataset.

## Preprocessing (as in the notebooks)

- Read with OpenCV (BGR channel order), resized to 128 x 128, scaled to [0, 1].
- Shuffled with seed 2, then split 90% / 10% (train / test) at random, without stratification. In the notebooks this gave 590 (Cassava), 48 (Rice) and 421 (Maize) test images.

## Sources

Please verify the exact source and license of each copy you use. The notebooks do not record where the data came from beyond folder names.

- **Cassava:** a cassava leaf disease image dataset with the classes cbb, cbsd, cgm, cmd and healthy (for example the Makerere AI Lab cassava data used in the iCassava challenge; see Mwebaze et al., *iCassava 2019 Fine-Grained Visual Categorization Challenge*, arXiv:1908.02900).
- **Rice:** a rice leaf diseases dataset with bacterial leaf blight, brown spot and leaf smut.
- **Maize:** a maize leaf disease dataset with blight, common rust, gray leaf spot and healthy.

Add the exact source and license of the copies you used before publishing results.
