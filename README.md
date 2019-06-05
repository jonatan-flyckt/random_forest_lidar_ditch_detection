# Ditch detection using refined LiDAR data
A bachelor's thesis in computer science at Jönköping University, in collaboration with the Swedish Forest Agency.

The code in this repository runs the algorithms from the bachelor's thesis *Ditch detection using refined LiDAR data* by Filip Andersson and Jonatan Flyckt

## Jupyter notebooks:
### random_forest_experiment
Displays the experiment and results for the random forest model.

### recreated_non_ML_experiment
Displays the experiment and results for the recreated non-machine learning method by Gustavsson and Selberg (2018)

### feature_creation
Displays the creation of the different features used in the random forest experiment.

## Functions:
### general_functions
Code used in several parts of the experiments.

### feature_creation
Code used for creating the 81 different features used in the random forest experiment.

### post_processing
Code used to process the ditch prediction after receiveing the output from the random forest model.


## Thesis abstract:
In this thesis, a method for detecting ditches using digital elevation data derived from LiDAR scans was developed in collaboration with the Swedish Forest Agency.

The objective was to compare a machine learning based method with a state of the art automated method, and to determine which LiDAR-based features represent the strongest ditch predictors.

This was done by using the digital elevation data to develop several new features, which were used as input in a random forest machine learning classifier. The output from this classifier was processed to remove noise, before a binarisation process produced the final ditch prediction. Several metrics including Cohen's Kappa index were calculated to evaluate the performance of the method. These metrics were then compared with the metrics from the results of a reproduced state of the art automated method.

The confidence interval for the Cohen's Kappa metric for the population was calculated to be **[0.567 , 0.645]** with a **95 %** certainty. Features based on the Impoundment attribute derived from the digital elevation data overall represented the strongest ditch predictors.

Our method outperformed the state of the art automated method by a high margin. This thesis proves that it is possible to use AI and machine learning with digital elevation data to detect ditches to a substantial extent.


**Filip Andersson** - anfi1622@student.ju.se - https://github.com/FilipAndersson245  
**Jonatan Flyckt** - fljo1589@student.ju.se - https://github.com/jonatan-flyckt
