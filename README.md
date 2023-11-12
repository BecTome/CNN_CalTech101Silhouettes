# CNN_CalTech101Silhouettes

## Download the dataset
```bash
bash run.sh
```

The format is `.mat` for MatLab matrices. However, can be read in python

```python
import scipy.io as io
data = io.loadmat('caltech101_silhouettes_28_split1.mat')
```

## Experiments Run

In [experiments](experiments) folder, a new folder is created for each experiment. Inside there should be a script which runs the experiment and saves the results in a folder called `results`.