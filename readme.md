## Introduction
*xSVM_ctx-Lib* is an open source library that implements algorithms proposed to create a parallel arrangement of Support Vector Machines (SVM) by contexts, where each SVM is trained in a specific context identified during training.

## Requirements
*xSVM_ctx-Lib* requires Python 3.8+. This library implements the [*XSVMC-Lib*](https://github.com/interpretapple-lab/xsvmc-lib) library for making contrasting explanations. It is necessary to install *XSVMC-Lib* in order to use this library. Moving the *xsvmlib* directory from *XSVMC-Lib* inside this directory (*xSVM_ctx-Lib*) will count as installed. 

Additionally, it is necessary to have the following packages installed:

- [Numpy](https://numpy.org) (```python3 -m pip install numpy```)
- [Matplotlib](https://matplotlib.org) (```python3 -m pip install matplotlib```)
- [Pandas](https://pandas.pydata.org) package (```python3 -m pip install pandas```).
- [Seaborn](https://seaborn.pydata.org) package (```python3 -m pip install seaborn```).
- [SciKit-Learn](https://scikit-learn.org) package (```python3 -m pip install scikit-learn```).
- [OpenCV](https://opencv.org) (```python3 -m pip install opencv-python```)
- [Joblib](https://joblib.readthedocs.io) package (```python3 -m pip install joblib```).

## Examples
To run an example, say *bookings.py*, you may use the following commands:

```
cd /path/to/xsvm_ctx-lib/examples/
python3 bookings.py
```

## Datasets
Parts of the following datasets are used in several examples that illustrate the use of *xSVM_ctx-Lib*.

- [The HAM10000 dataset, images of common pigmented skin lesions](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) (This dataset was preprocessed in advanced due to its size.)
- [EMG Data for Gestures](https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures)
- [Hotel booking demand dataset](https://www.sciencedirect.com/science/article/pii/S2352340918315191)

Examples
To run an example, say bookings.py, you may use the following commands:

```
cd /path/to/xsvm_ctx-lib/examples/
python3 bookings.py
```

## Technical Information
The mathematical foundation of *xSVM_ctx-Lib* can be found in 

M. Loor, A. Tapia-Rosero and G. De Tré. "[Contextual Boosting to Explainable SVM Classification](https://doi.org/10.1007/978-3-031-39965-7_40)" Massanet, S., Montes, S., Ruiz-Aguilera, D., González-Hidalgo, M. (eds) Fuzzy Logic and Technology, and Aggregation Operators. EUSFLAT AGOP 2023 2023. Lecture Notes in Computer Science, vol 14069. Springer, Cham. https://doi.org/10.1007/978-3-031-39965-7_40

## License
*xSVM_ctx-Lib* is released under the [Apache License, Version 2.0](LICENSE).

## Citing
If you use *xSVM_ctx-Lib*, please cite the following article:

M. Loor, A. Tapia-Rosero and G. De Tré. "[Contextual Boosting to Explainable SVM Classification](https://doi.org/10.1007/978-3-031-39965-7_40)" Massanet, S., Montes, S., Ruiz-Aguilera, D., González-Hidalgo, M. (eds) Fuzzy Logic and Technology, and Aggregation Operators. EUSFLAT AGOP 2023 2023. Lecture Notes in Computer Science, vol 14069. Springer, Cham. https://doi.org/10.1007/978-3-031-39965-7_40

### BibTeX
```
@INPROCEEDINGS{
    xsvm@ctx,
    author={Loor, Marcelo and Tapia-Rosero, Ana and De Tr{\'e}, Guy},
    editor={Massanet, Sebastia and Montes, Susana and Ruiz-Aguilera, Daniel and Gonz{\'a}lez-Hidalgo, Manuel},
    title={Contextual Boosting to Explainable SVM Classification},
    booktitle={Fuzzy Logic and Technology, and Aggregation Operators},
    year={2023},
    publisher={Springer Nature Switzerland},
    address={Cham},
    pages={480-491},
    isbn={978-3-031-39965-7},
    doi={10.1007/978-3-031-39965-7_40}
}

```
