__author__ = "Tanay Rastogi"
"""
The puprose of the file is to load the dataset in a manner that can be used for training a ANN in Pytorch. 
Class ModeChoice implements the Pytorch's CustomDataset class to 
     - load data from the disk.
     - transform input and output data in form required.  
     - 

REFERENCE
 - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
 - https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/
"""

import pandas as pd
import torch 
import torch.utils.data as torchdata
torch.manual_seed(42)
import pickle


def fetch_train_test_data(data, dataset_type="train"):
    if dataset_type == "train":
        with open('train_ids.pkl', 'rb') as f:
            train_idx = pickle.load(f)
            return data.iloc[train_idx]

    elif dataset_type == "test":
        with open('test_ids.pkl', 'rb') as f:
            test_idx = pickle.load(f)
            return data.iloc[test_idx]
    else:
        raise ValueError("Wrong dataset type!")



class ModeChoiceDataset(torchdata.Dataset):
    """    Class to load the ModeChoice data from CSV    """
    def __init__(self, csv_path, data_type="train", normalize=False, verbose=False) -> None:
        # Read CSV
        data = pd.read_csv(csv_path, delimiter=";", header=0, decimal=".", index_col="id")
        data = fetch_train_test_data(data, dataset_type=data_type)

        ### Features tensor ### 
        not_feature = ["mode", "car_ok", "pass_ok", "bus_ok", "train_ok", "walk_ok", "bike_ok"]
        features = data.loc[:, [col for col in data.columns if col not in not_feature]]
        self.X = features.to_numpy()                                    # n_samples x feature_len
        # Normalize features columns-wise
        if normalize:
            # Min-Max scaler
            self.X -= self.X.min(axis=0)
            self.X /= self.X.max(axis=0) - self.X.min(axis=0)
        self.X = torch.from_numpy(self.X)
        self.X = self.X.type(torch.FloatTensor)                                    # n_samples x feature_len
        
        ### Label tensor ### 
        # Labels converted to one_hot
        self.label_ind = data.loc[:, "mode"].to_numpy() - 1                        # n_samples
        self.label_ind = torch.from_numpy(self.label_ind)
        self.y = self.label_ind.type(torch.long)

        # Label dict for visulization during analysis
        self.labels = {0: "Car Driver",
                       1: "Car Passenger",
                       2: "Bus", 
                       3: "Train",
                       4: "Walk",
                       5: "Bike"}

        # Name of the features in the data
        self.features_name = list(features.columns)

        # Simle Printing
        if verbose:
            print("\n--------- ModeChoice {}-Dataset ---------".format(data_type.upper()))
            print("Number of datapoints:", self.X.shape[0])
            print("Feature Length:", self.X.shape[1])
            print("Label   Length:", len(self.labels))
            print("----------------------------------------")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    

if __name__ == "__main__":

    import plotly.graph_objects as go
    ### This is only for visualization purposes of dataset ###
    # The script is mostly to use as a dataset class for training and testing ANN 

    # Labels for the modes
    labels = {0: "No Mode",
                1: "Car Driver",
                2: "Car Passenger",
                3: "Bus", 
                4: "Train",
                5: "Walk",
                6: "Bike"}

    ## Loading Data ##
    path = "data/modeData.csv"
    data = pd.read_csv(path, delimiter=";", header=0, decimal=".", index_col="id")
    data = fetch_train_test_data(data, dataset_type="test")
    
    ## Bar plot for the imbalance in dataset ##
    # Count of each type of mode
    count_data = data["mode"].value_counts()
    # Plot
    bar = go.Figure()
    bar.add_trace(go.Bar(x=[labels[idx] for idx in count_data.index],
                         y=count_data.values,
                         text=count_data.values, textposition="auto"))
    bar.update_xaxes(title_text='Available Modes')
    bar.update_yaxes(title_text='Count')
    bar.update_layout(title="Imbalance in Dataset", title_x=0.5)
    bar.update_layout(margin={"r":20,"t":50,"l":20,"b":20})
    bar.show()