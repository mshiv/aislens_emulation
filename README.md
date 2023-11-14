AISLENS Emulation
==============================

This project includes code/workflows for a statistical generator of spatiotemporal variability in forcings for ice sheet models. The generator, as applied to ocean model simulation output used to force an ice sheet model, has been described in detail in [[1]](#1).

Statistical Emulation of Antarctic Ice Sheet Melt Projections

# Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>

--------
# Workflow Description

## Requirements
#### Dependencies:
Python environment dependencies/package list is provided in [_environment.yml_](https://github.com/mshiv/aislens_emulation/blob/main/environment.yml).

#### Data:

A processed version of the data used in this workflow is made available as part of the [supplementary files](https://zenodo.org/doi/10.5281/zenodo.7633996) published with [[1]](#1).
<p><small>E3SM MPAS-O data output files: _mpaso.hist.am.timeSeriesStatsMonthly.{timeStamp}.nc_. These are made available on the remote cluster.</small></p>


Analysis documentation to be provided as Sphinx docs.

## References
<a id="1">[1]</a>
S. Muruganandham, A. A. Robel, M. J. Hoffman and S. F. Price, "[Statistical Generation of Ocean Forcing With Spatiotemporal Variability for Ice Sheet Models](https://ieeexplore.ieee.org/document/10201387)," in Computing in Science & Engineering, vol. 25, no. 3, pp. 30-41, May-June 2023, doi: 10.1109/MCSE.2023.3300908.

