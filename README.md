# Task-Optimal Canonical Correlation Analysis

Canonical Correlation Analysis (CCA) is widely used for multimodal data analysis and, more recently, for discriminative tasks such as multi-view learning; however, it makes no use of class labels. Recent CCA methods have started to address this weakness but are limited in that they do not simultaneously optimize the CCA projection for discrimination and the CCA projection itself, or they are linear only. We address these deficiencies by simultaneously optimizing a CCA-based and a task objective in an end-to-end manner. Together, these two objectives learn a non-linear CCA projection to a shared latent space that is highly correlated and discriminative. Our method showed a significant improvement over previous stateof-the-art (including deep supervised approaches) for cross-view classification, regularization with a second view, and semi-supervised learning on real data.

## Setup

Basic installation requires a number of python packages, which are most easily installed with conda:

```
conda install -c conda-forge numpy scipy cudatoolkit cudnn keras tensorflow-gpu scikit-learn
```

## Usage

Example usage on MNIST split with random search for hyperparameter optimization:
```
python run_tocca.py --training_size 10000 -m w -p 50   # TOCCA-W
python run_tocca.py --training_size 10000 -m sd -p 50  # TOCCA-SD
python run_tocca.py --training_size 10000 -m nd -p 50  # TOCCA-ND
```


## Citation

If you use this code, please cite:

```
@article{couture2019tocca,
  author    = {Heather D. Couture and
               Roland Kwitt and
               J. S. Marron and
               Melissa A. Troester and
               Charles M. Perou and
               Marc Niethammer},
  title     = {Deep Multi-View Learning via Task-Optimal {CCA}},
  year      = {2019},
  url       = {http://arxiv.org/abs/1907.07739},
  archivePrefix = {arXiv},
  eprint    = {1907.07739},
}
```
