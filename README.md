# Mean-Variance Inference Under Gaussian Process for Robust Bayesian Optimization 

This repository contains the code and experiments done in the work ["Spectral Representation of Robustness Measures for Optimization Under Input Uncertainty"](https://icml.cc/virtual/2022/spotlight/17742) by Jixiang Qing, Tom Dhaene and Ivo Couckuyt.

![](https://github.com/TsingQAQ/Image-Lib/blob/master/figure_for_pwebsite/QFF_vs_RFF.png?raw=true)
--------------

### Installation
:exclamation::exclamation::exclamation:Caution: You are away from [the main branch of Trieste](https://github.com/secondmind-labs/trieste), this branch contains certain other dependencies  

install from sources, run
```bash
$ pip install -e.
```
in the repository root (tested with Python version 3.7.10).  

--------------

### Tutorial Notebook
There is a tutorial notebook `robust_optimization_considering_mean_variance.pct.py` at (`\docs\notebooks`) demonstrating: 
1) how to make use of the mean and variance inference.
2) how to use them for robust Bayesian Optimization.  
  
  In order to run the notebook, install the following dependency:     
  ``` 
  $ pip install -r notebooks/requirements.txt
  ```  

  Then, run the notebooks with  
   ```
  $ jupyter-notebook notebooks
  ```
  --------------
  
### Reproduce the paper's result
If you'd like to reproduce the paper's result exactly, the following directories contain relevant experiments:

- `docs\exp\FF_Variance\uncertainty_calibration` Uncertainty Calibration
- `docs\exp\FF_Variance\mc_comparison_of_input_and_spectral_density` First Moment Comparison 
- `docs\exp\FF_Variance\robust_bayesian_optimization_exp` RBO experiments
  - `\scalar_mean_var_exp`
  - `\var_as_con_acq_exp`
  - `\mo_mean_var_exp`
  
Note: some scripts containing plot labels that depends on a local LaTeX compiler. 

--------------
### Citation

If you find this work or repository helpful, please kindly consider citing our work:
```
to be appear soon
```
