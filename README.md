# Heterogeneous BOTL

Source code & experiments for:

A. Deshwal, S. Cakmak, Y. Xia and D. Eriksson. Sample-Efficient Bayesian Optimization with Transfer Learning for Heterogeneous Search Spaces. AutoML Conference 2024 (Workshop Track).

[Link to OpenReview & paper PDF](https://openreview.net/forum?id=FSqIx6FO0O)

```
@inproceedings{
deshwal2024sampleefficient,
title={Sample-Efficient Bayesian Optimization with Transfer Learning for Heterogeneous Search Spaces},
author={Aryan Deshwal and Sait Cakmak and Yuhou Xia and David Eriksson},
booktitle={AutoML Conference 2024 (Workshop Track)},
year={2024},
}
```

# Get Started

The code was developed using Python 3.10. The dependencies are listed in `requirements.txt` with fixed versions, and will be installed upon execution of the commands below.

```
pip install -e .
pytest -ra
```

# Running benchmarks

There are three benchmarks in `heterogeneous_botl/benchmarks` directory named `hartmann_bo_loop.py`, `ranger_hpob_bo_loop.py`, `rpart_hpob_bo_loop.py`. Each file runs four different methods `random`, `single-task`, `het_mtgp`, `imputed_mtgp`, `common_mtgp`, `learned_imputed_mtgp` defined in `heterogeneous_botl/helpers` using models defined in `heterogeneous_botl/models`.

NOTE: To run HPO-B benchmarks, the datasets must be downloaded and extracted to `heterogenous_botl/benchmarks/hpo_b/hpob-data` directory. You can use the download link [HERE](
https://rewind.tf.uni-freiburg.de/index.php/s/xdrJQPCTNi2zbfL/download/hpob-data.zip). You can refer to the [HPO-B GitHub repo](https://github.com/releaunifreiburg/HPO-B) in case of any issues.

To run a benchmark:

```python benchmarks/[benchmark_name] [arguments]```

Examples (use defaults for the main results in the paper):

```
python heterogeneous_botl/benchmarks/hartmann_bo_loop.py  --fixed_constant 0.0 --imputation_value 0.5 --n_bo_steps 35 --n_source_samples 30 --n_replications 10 --n_init_target_samples 5
```

```
python heterogeneous_botl/benchmarks/ranger_hpob_bo_loop.py  --train_dataset_index 0 --test_dataset_index 0 --n_bo_steps 35 --n_source_samples 30 --n_replications 10 --n_init_target_samples 5
```

```
python heterogeneous_botl/benchmarks/rpart_hpob_bo_loop.py  --train_dataset_index 0 --test_dataset_index 0 --n_bo_steps 35 --n_source_samples 30 --n_replications 10 --n_init_target_samples 5
```

The benchmark outputs will be saved under `results/` directory.


### Reproducing plots from the paper

The notebooks `results/botl_results.ipynb` and `results/ranger_ablation.ipynb` will produce the figures presented in the paper using the included benchmark results.


### Common arguments to all benchmarks:

`n_bo_steps` - No. of BO steps in each replication.

`n_source_samples` - Size of source task datasets.

`n_init_target_samples` - No. of init target samples.

`n_replications` - No. of replications.


### Arguments specific to Hartmann:

`fixed_constant` - Fixed value to use for unobserved features to evaluate the objective.

`imputation_value` - Fixed imputation value for imputed_mtgp


#### Arguments specific to HPO-B benchmarks (ranger and rpart):

`train_dataset_index` - Index of training (source) dataset from HPO-B (values between 0 to 50)

`test_dataset_index` - Index of testing (target) dataset from HPO-B (values in {0, 1} for ranger and in {0, 1, 2, 3} for rpart)


## License
Heterogeneous BOTL is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.
