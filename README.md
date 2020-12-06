# Supervised machine learning for automated classification of human Wharton’s Jelly cells and mechanosensory hair cells

Code accompanying the paper "Supervised machine learning for automated classification of human Wharton’s Jelly cells and mechanosensory hair cells"

All code was developed in macOS using Python 3.6.5. For all other information regarding versions of required packages, check the “requirements.txt” file.


## Data

The cell features used for this project can be found in data/cell_data.csv. These features were computed using images collected from [[1]]. Further details regarding the process used to collect the data can be found in the paper. 

[1]: https://github.com/AbihithK/HWJC_MHC_Classification/blob/master/README.md#references

## Usage

To replicate any of the models found in the paper, simply navigate to the directory and execute:

	python main.py --model [MODEL]

where [MODEL] can be “l1’ for the L1-regularized logistic regression, “l2” for the L2-regularized logistic regression, “svm” for the support vector machine, “knn” for the k-nearest neighbors model, “rf” for the random forest model, “mlp” for the multi-layer perceptron, or “vc” for the voting classifier model.



To run the script used to identify active features, execute:
	
	python main.py --active_features

This will output the weights of all identified active features, and save the data to data/cell_data_active_features.csv. This file can now be used to develop the models again, with only the active features, by executing:

	python main.py --data_file cell_data_active_features.csv --model [MODEL]



In the paper, median performance metrics from across training iterations were reported. To instead report the max performance metrics across the training iterations, execute:

	python main.py --model [MODEL] --evaluation_mode max

If used when saving the model, the model that produced the max performance will be saved. More information on saving models can be found below. 





### Usage with other data

The models used in the paper can also be used on other data. To do so, simply place the data file in the data/ folder. Then execute:

	python main.py --data_file [FILE] --target_feature [FEATURE] --model [MODEL]

where [FILE] is the name of the data file and [FEATURE] is the name of the target feature.



To identify active features with your data, execute:

	python main.py --data_file [FILE] --target_feature [FEATURE] --active_features





### Checkpoints

To periodically save checkpoints during training, execute:

	python main.py --model [MODEL] --save_checkpoints --checkpoint_save_frequency [FREQUENCY]

This will save a checkpoint every [FREQUENCY] runs to checkpoints/checkpoint.json. Training can then be resumed from this checkpoint using:

	python main.py --load_checkpoint





### Saving and loading models

To save the model that produced the median or max AUC score (depending on the evaluation mode), execute:

	python main.py --model [MODEL] --save_model 

This will save a .json file and a .joblib file in models/trained/. To load the model later for evaluation or prediction on other data, execute:

	python main.py --data_file [FILE1] --load_model [FILE2]

where [FILE2] is the name of the .json file. This will output the AUC/accuracy scores if used for evaluation with [FILE1], or it will save its predictions on [FILE1] in output/pretrained_predictions/, depending on whether the data has already been classified.



NOTE: The --save_model option will not work with the --save_checkpoints option. Executing both arguments at once will only override the --save_checkpoints option.





### Other inputs

For other inputs, refer to

	python main.py --help


## ROC curves

To display ROC curves, execute:
	
	python roc.py

This will automatically display a plot of all ROC curves of previously trained models, using the .csv files in the output/ folder. Each curve will be displayed in a random color. To specify which colors to use, execute:

	python roc.py --colors [COLOR1] [COLOR2] [COLOR3]...

where [COLOR1], [COLOR2], [COLOR3], ... are all matplotlib colors. 

If the .csv files in the output/ folder were developed using data other than the data provided, execute:

	python roc.py --positive_label [LABEL]

where [LABEL] is the name of the positive label of the target feature.


## Help

Feel free to contact abihith.kothapalli@gmail.com if you have any questions or problems running these scripts.

## References

<a id="1">[1]</a> Mellott AJ, Devarajan K, Shinogle HE, Moore DS, Talata Z, Laurence JS, et al. Nonviral Reprogramming of Human Wharton's Jelly Cells Reveals Differences Between ATOH1 Homologues. Tissue Eng Part A. 2015;21(11-12):1795-809. Epub 2015/03/12. doi: 10.1089/ten.TEA.2014.0340. PubMed PMID: 25760435; PubMed Central PMCID: PMCPMC4449705.
