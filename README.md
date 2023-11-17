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

```bash
python train.py --name first_run --lr 0.00005 --bs 128 --ep 20 --partition "80_10_10"
```

## Gradcam

In addition to that, explainability techniques such as GradCAM are implemented in [gradcam.py](gradcam.py). The results are saved in a folder called `gradcam`. Where `nclass` is the number of the class whose GradCAM is to be visualized.

```bash
python gradcam.py --model "path/to/model.h5" --nclass 100 --name gradcam
```