# MNL v sANN
Project to compare MNL model with ANN model for a mode choice data. Both the model are trained on the same data and compared on accuracy of predictions. 

## REPO Structure
In order to run the code base you will need the following Python libraries - 
 - Numpy
 - Torch
 - Scikit Learn
 - Plotly

The Mode Choice data is stored in *data/modeData.csv*. Files *train_ids.pkl* and *test_ids.pkl* save the ids for train and test data, such that train/test split have the same ratio of classes in both. File *dataset.py* creates the implements the custom dataset used for PyTorch implementation. 

### MNL
All the MNL code is in the jupyter notebook. The notebook implements the training and testing of MNL model. The model that is trained is also explained within the notebook. 

### ANN
We train and test a multi-layer perceptron with differnt hidden layers. File *HyperParameterSearch.py* implements the different parameters that we tested to check the best model. File *models.py* implements the ANN models with dynamic number of layers. 

The model is varies on, 
 - Number of hidden layers.
 - Batch size for training.
 - Number of epochs for training.
 - Learning rate for optimizer.
 - Different type of optimizers.




