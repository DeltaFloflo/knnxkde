# $`k`$NN$`\times`$KDE

GitHub repository for the $`k`$NN$`\times`$KDE algorithm.

## Description

**Numerical Data Imputation for Multimodal Data Sets:  A Probabilistic Nearest-Neighbor Kernel Density Approach**  
TMLR, June 2023

All methods and data are available here.  
You can reproduce all results and figures with the Jupyter Notebooks.

## Structure

Folders

* `GAIN/` Original implementation of Generative Adversarial Imputation Nets (Yoon et al.)
* `data/` 15 datasets in csv format (12 real datasets + 3 synthetic 2D datasets)
* `figures/` Figures presented in the manuscript (you can reproduce them)
* `output/` Exhaustive results (can be used to reproduce the Tables and Figures)
* `softimpute/` Original implementation of the matrix completion SoftImpute algorithm

Files

* `compute_likelihood.py` Compute log-likelihood for $`k`$NN$`\times`$KDE, $`k`$NN-Imputer, MissForest, MICE, and Mean. Save results for the log-likelihood scores. Uses best hyperparams from the RMSE results. Need to specify the missing data scenario ('full\_mcar', 'mcar', 'mar', 'mnar')
* `demo.ipynb` **[START HERE]** Friendly Jupyter notebook to demonstrate how to use the $`k`$NN$`\times`$KDE with your own dataset. 
* `hyperparam_rmse.py` Compute the NRMSE for all datasets, with 6 different missing rates, on a grid of parameters for each method. Save results for the NRMSE. Need to specify the missing data scenario ('full\_mcar', 'mcar', 'mar', 'mnar')
* `knnxkde.py` Implementation of the $`k`$NN$`\times`$KDE.
* `make_appendix.ipynb` Jupyter notebook to reproduce the figures in the Appendix of the manuscript.
* `make_other_figures.ipynb` Jupyter notebook to reproduce Figures 1, 2, and 3 from the main text of the manuscript.
* `notes.txt` Blueprint of the methodology of the manuscript
* `probabilistic_standard_methods.py` Devise a way to obtain a likelihood with standard imputation methods. Used subsequently to compute the log-likelihood.
* `rmse_likelihood_results_plots.ipynb` Jupyter notebook to reproduce Figures 4 and 5 in the main text of the manuscript. Generates the scripts for the LaTex tables.
* `tests_movielens_dataset.ipynb` Requested by reviewer.
* `utils.py` Utils functions.

## Citation

```bibtex
@article{lalande2023,
  title={Numerical Data Imputation for Multimodal Data Sets: A Probabilistic Nearest-Neighbor Kernel Density Approach},
  author={Florian, Lalande and Kenji, Doya},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2023},
  url={https://openreview.net/forum?id=KqR3rgooXb},
  note={Reproducibility Certification},
}
