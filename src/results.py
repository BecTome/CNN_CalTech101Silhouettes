import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import src.config as config
import os

def plot_history(model):
    
    df_hist = model.history.history
    df_hist = pd.DataFrame(df_hist)

    fig, ax = plt.subplots(1,2, figsize=(15,5))
    df_hist[['accuracy', 'val_accuracy']].plot(ax=ax[0])
    df_hist[['loss', 'val_loss']].plot(ax=ax[1])

    ax[0].set_ylabel('Accuracy')
    ax[0].set_ylim([0,1])
    ax[1].set_ylabel('Loss')

    ax[0].set_xlabel('Epoch')
    ax[1].set_xlabel('Epoch')
    return fig

def plot_history_logloss(model):
    
    df_hist = model.history.history
    df_hist = pd.DataFrame(df_hist)

    fig, ax = plt.subplots(1,2, figsize=(15,5))
    df_hist[['accuracy', 'val_accuracy']].plot(ax=ax[0])
    df_hist[['loss', 'val_loss']].plot(ax=ax[1])

    ax[0].set_ylabel('Accuracy')
    ax[0].set_ylim([0,1])
    ax[1].set_ylabel('Loss')
    ax[1].set_yscale('log')
    
    ax[0].set_xlabel('Epoch')
    ax[1].set_xlabel('Epoch')
    return fig

def get_cm(y_true, y_pred, pct=False):
    cm = confusion_matrix(y_true, y_pred)
    if pct:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm

def plot_cm(y_true, y_pred, labels, pct=False):
    import seaborn as sns
    cm = get_cm(y_true, y_pred, pct=pct)
    fig = plt.figure(figsize=(30, 15))
    if pct:
        sns.heatmap(cm, annot=False, fmt='.2f',
                    xticklabels=labels, yticklabels=labels)
    else:
        sns.heatmap(cm, annot=False, fmt='d',
                    xticklabels=labels, yticklabels=labels)
        
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion matrix')
    return fig

def class_report(y_true, y_pred, out_path=None, labels=config.LABELS):
    report = metrics.classification_report(y_true, y_pred, target_names=labels)
    if out_path is not None:
        with open(os.path.join(out_path, 'classification_report.txt'), 'w') as f:
            f.write(report)
    
    print(report)
    return report

def save_params(out_path, **kwargs):
    import json
    with open(os.path.join(out_path, 'params.json'), 'w') as f:
        json.dump(kwargs, f, indent=4)

def write_summary(model, out_path):
    with open(os.path.join(out_path, 'summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

def generate_report(model, X_val, y_val, output_path):

    y_prob = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    # fig = plot_cm(y_val, y_pred, labels=config.LABELS)
    # fig.savefig(os.path.join(output_path, 'confusion_matrix.jpg'))

    fig = plot_cm(y_val, y_pred, labels=config.LABELS, pct=True)
    fig.savefig(os.path.join(output_path, 'confusion_matrix_pct.jpg'))

    report = class_report(y_val, y_pred, out_path=output_path)