# Accompanying Plots

Plots and assocated data are presented within this folder.

As December 2021, the plotting code requires installation of [`seaborn`](https://seaborn.pydata.org/) directly from GitHub to [plot error bars](https://github.com/mwaskom/seaborn/issues/2403) with two standard deviations. Installing `seaborn` from a earlier release (v 0.11) will not allow these plots to run withe appropriate error bars. 

To run the plotting code, install the dependencies with pip:

```
pip install -r requirements.txt
```

We have provided the plots as both a Python script, [`Plot_Notebook.py`](https://github.com/pasta41/deception/blob/main/src/plot/Plot_Notebook.py) and as a Jupyter notebook, [`Plot_Notebook.ipynb`](https://github.com/pasta41/deception/blob/main/src/plot/Plot_Notebook.ipynb).  The script is a copy of the notebook eexported as a Python script.
