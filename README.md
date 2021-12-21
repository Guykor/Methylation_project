# Methylation_project

## Description
This final DS project contains development of mixture model for cfDNA deconvolution problem with DNA methylation data (that is not supplied here since its medical confidential data).
A complete report, including development of the algorithms, is included in this repo.
Elements:
1. The project includes Loader class, which is in charge of loading the required elements (UMX counts profiles comprising the atlas, divided to train and test, Features or marker file, whole blood samples and some more cfDNA profiles, and initial vector theta for relative contribution).
the object is generic loader for count profiles that handles alignment and subseting the problem.
2. Atlas object
3. Theta object.
4. Evaluators.
5. experiment config.

## Getting Started
1. Get data from Prof. Tommy Kaplan's Lab
2. Clone repo and handle paths in experiment config.
3. define Experiment and run. 

## Example:
Complete example of running a full experiment can be found in experiment.py file.

The following code loads all relevant data, subsets and align indices of markers and tissues (existence and ordering)
```python
# define loading procedure - enforces same format (markers and tissue) to all data used.
from data_api import Loader
from experiment_config import config
loader = Loader(config)
```

The code allows aggregation of  tissues, with choice of subseting also their markers from all the objects thus changing the whole experiment problem.
```python
# define loading procedure - enforces same format (markers and tissue) to all data used.
loader = Loader(config)

# By markers alows aggregation that emphasises the data from relevant markers of each cell type
loader.unify_tissues('Pancreas', ['Pancreas-Alpha', 'Pancreas-Beta', 'Pancreas-Delta'], by_markers=True)

# Subset experiment (Select cell types for theta object, train and test atlas and marker relevance table
loader.subset_tissues(['Bladder-Ep', 'Liver-Hep', 'Lung-Ep-Alveo', 'Lung-Ep-Bron', 'Neuron'], relevant_markers=False)

# Subset only markers from data artifacts
liver_rel_markers = (loader.markers_importance['Liver-Hep'] != 0).index.values
loader.subset_markers(liver_rel_markers)

# Specific Grouping method for white blood cells
loader.group_wbc(config.WBC_NAME, by_markers=True)
```

To test validity of the process, one can define a logging mechanism that outputs in a file.
```python
from project_logger import setup_logger, FILE_HANDLER
logger = setup_logger()
```

# Plotting Report Figures example:
from within report_figures directory.
```
python -c from plots.py import generate_figure_2_1; generate_figure_2_1
```
