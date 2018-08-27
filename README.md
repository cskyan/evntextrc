# Biological Event Extractor

`EvntExtrc` is a biological event extractor on biomedical literature. It can be used to evaluate the performance of different methods on various event detection tasks. The tool firstly transforms the unstructured textual data into numerical or binary feature matrix which is further used for cross-validation or prediction. The tool has implemented a general interface for feature construction and prediction post-processing from different datasets. The recommended model `VecEntNet` uses bi-directional LSTM to model the context of arguments in each event. The argument embeddings are then used to train the directed event detection model `VeComNet`. The trained models can be adopted for predictions of biomedical events from the text with [Entity annotation](http://2011.bionlp-st.org/home/file-formats#TOC-Entity-annotation-files-.a1-). It can generate the corresponding files with [Event annotation](http://2011.bionlp-st.org/home/file-formats#TOC-Event-annotation-files-.a2-). It also provides several utility functions to manipulate the dataset and post-process the results.

## Getting Started

The following instructions will help you get a copy of the source code as well as the datasets, and run the programs on your own machine.

### Prerequisities

Firstly, you need to install a Python Interpreter (tested 2.7.12) and these packages:

* numpy (tested 1.14.2)
* scipy (tested 1.0.0)
* matplotlib (tested 2.2.2)
* pandas (tested 0.20.3)
* scikit-learn (tested 0.19.1)
* keras (tested 2.1.5)
* gensim (tested 3.0.1)
* optunity (tested 1.1.1)
* pyyaml (test 3.12)
* openpyxl (test 2.4.9)

The simplest way to get started is to use [Anaconda](https://www.continuum.io/anaconda-overview) Python distribution. If you have limited disk space, the [Miniconda](http://conda.pydata.org/miniconda.html) installer is recommended. After installing Miniconda and adding the path of folder `bin` to `$PATH` variable, run the following command:

```bash
conda install scikit-learn pandas matplotlib optunity openpyxl
```

### Download the Source Code

You can clone the repository of this project and then update the submodule after entering the main folder:

```bash
git clone https://github.com/cskyan/evntextrc.git
cd evntextrc
git submodule update --init --recursive
```

Or you can clone the repository and submodules simultaneously:

```bash
git clone --recursive https://github.com/cskyan/evntextrc.git
```

### Configure Environment Variable

* Add the path of folder `bin` to `$PATH` variable so that you can run the scripts wherever you want. *Remember to grant execution permissions to all the files located in* `bin`
* Add the path of folder `lib` to `$PYTHONPATH` variable so that the Python Interpreter can find the library `bionlp`.

### Configuration File

The global configuration file is stored as `etc/config.yaml`. The configurations of different functions in different modules are separated, which looks like the code snippet below.

```
MODULE1:
- function: FUNCTION1
  params:
    PARAMETER1: VALUE1
    PARAMETER2: VALUE2
- function: FUNCTION2
  params:
    PARAMETER1: VALUE1
	
MODULE2:
- function: FUNCTION1
  params:
    PARAMETER1: VALUE1
```

Hence you can access a specific parameter VALUE using a triple (MODULE, FUNCTION, PARAMETER). The utility function `cfg_reader` in `bionlp.util.io` can be used to read the parameters in the configuration file:

```python
import bionlp.util.io as io
cfgr = io.cfg_reader(CONFIG_FILE_PATH)
cfg = cfgr(MODULE, FUNCTION)
VALUE = cfg[PARAMETER]
```

The parameters under the function `init` means that they are defined in module scope, while the parameters under the function `common` means that they are shared among all the functions inside the corresponding module.

### Locate the Pre-Generated Dataset

After cloning the repository, you can download some pre-generated datasets [here](https://data.mendeley.com/datasets/mr39zgc7y5) . The datasets described below are organized as `pandas` DataFrame stored in [`HDF5`](https://portal.hdfgroup.org/display/HDF5/HDF5) data model using the [`to_hdf`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_hdf.html) function of `pandas`. 

Filename | Description  
--- | ---
dataset.h5 | Pre-processed dataset for a task in BioNLP Shared Task
test_rawdata.pkl | Textual data and entity annotations in the testing dataset
train_stat_oprnd_annot.pkl | Argument statistics for each event type in the training dataset
dev_stat_oprnd_annot.pkl | Argument statistics for each event type in the development dataset
pred_on_dev.zip | Prediction files with event annotation

HDF5 Key in `dataset.h5` | Description
cbow/[train\|dev\|test]_X[0-3] | Bi-directional word embedding ID streams constructed for the two arguments in each event
cbow/[train\|dev\|test]_ent_X[0-1] | Bi-directional word embedding ID streams constructed for the arguments
cbow/[train\|dev]_argvec[0-K]_X[0-1] | Bi-directional argument embedding for the arguments 0 to K
cbow/[train\|dev]_pseudo_X[0-1] | Bi-directional argument pseudo labels for each event
cbow/[train\|dev]_ent_Y | Argument labels
cbow/[train\|dev]_Y | Event labels


**The prefix of the HDF5 key represents the feature construction method (FeatureType). In order to locate the dataset you want to use, please change the parameter `DATA_PATH` of module `bionlp.spider.bnlpst` inside `etc/config.yaml` into the location of 'dataset.h5'.**

You can load a dataset into a [Pandas DataFrame](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html), with the corresponding `DocumentID|EntityID1|EntityID2` as index (e.g. BB-event-24831788|T9|T10) and each feature/label as column name, using the utility function `read_hdf` in `pandas` library with the FeatureType/Subset_Variable as parameter (e.g. cbow/train_Y, cbow/dev_X0, cbow/test_X1, etc.):

```python
import pandas as pd
Y = pd.read_hdf('dataset.h5', 'cbow/train_Y', iterator=False, chunksize=None)
```

### A Simple Example

You can run a demo using the following command:

```bash
evnt_extrc.py -m demo
```

*If your operating system is Windows, please use the Python Interpreter to execute the python scripts:*

```bash
python evnt_extrc.py -m demo
```

This demo will automatically download a pre-generated dataset and perform a 5-fold cross validation on each argument and event classification, followed by the prediction on development dataset based on the model trained on training dataset. The log is printed to standard output and the results are saved on the disk.

## Run a specific component
`EvntExtrc` mainly has two components, `cbow_ent` (VecEntNet), `cbow` (VeComNet). You can run each component using parameter `-e` to indicate the component name. For example:

```bash
evnt_extrc.py -e cbow_ent
```

## Parameter Tuning

For the sake of the best performance, you should tune the parameters of your selected model and write them on the model configuration file so that you can use these tuned parameters for model evaluation. 

### Setup parameter range

You can edit the function `gen_mdl_params` inside `bin/evnt_extrc.py` to change the range of parameter tuning. Please uncomment the code lines corresponded to your selected model and change the range of the parameters or append other values you want to test.

### Run parameter tuning script

You can choose an approach for parameter tuning using the following command.

*Particle Swarm Search*:

```bash
evnt_extrc.py -t -r particle_swarm
```

*Grid Search*:

```bash
evnt_extrc.py -t -r grid_search
```

*Random Search*:

```bash
evnt_extrc.py -t -r random_search
```

More details about the search methods (solvers) please refer to the documents of [Optunity](http://optunity.readthedocs.io/en/latest/user/solvers.html)

### Convert the result to configuration file

You can use the utility function in `bin/evnt_helper.py` to transformat your tuning result by the following command:

```bash
evnt_helper.py -m n2y -l TUNING_OUTPUT_FOLDER_PATH
```

**Then copy the basename of the configuration file ended with `.yaml` to the parameter `mdl_cfg` of module `evnt_extrc` inside `etc/config.yaml`.**

The pre-tuned parameters for some models are stored in `etc/mdlcfg.yaml`.

## Model Evaluation

You can use different combination of the feature selection/reduction model and classification model to generate a pipeline as the final computational model.

You can uncomment the corresponding code lines of the models you want to evaluate in functions `gen_featfilt`, `gen_clfs`, and `gen_clt_models` inside `bin/evnt_extrc.py` for feature selection/reduction and classification respectively.

In addition, you can use command line parameter `-c` to adopt the pre-combined model in functions `gen_cb_models` and `gen_cbclt_models`. To make use of the parameters stored in configuration file, you can use command line parameter `-c -b` to adopt the pre-combined model with optimized parameters.

## Dataset Re-Generation

You can re-generate the dataset from the annotated datasets [BioNLPST2011](http://2011.bionlp-st.org), [BioNLPST2013](http://2013.bionlp-st.org), [BioNLPST2016](http://2016.bionlp-st.org) stored in `DATA_PATH` using the following command:

```bash
evnt_gendata.py -m gen -y YEAR -u TASK
```

## Common Parameter Setting

* _-p [0-2]_  
specify which label you want to use independently
* _-l_  
indicate that you want to use all labels simultaneously
* _-k NUM_  
specify *K*-fold cross validation
* _-a [micro | macro]_  
specify which average strategy you want to use for multi-label annotation
* _-n NUM_  
specify how many CPU cores you want to use simultaneously for parallel computing

**Other parameter specification can be obtained using `-h`.**