__author__ = "Tanay Rastogi"
"""
File for hyperparameter search. 
This implements - training, validation and testing of model. 
We also plot the model and save it in folder - OUTPUT_FOLDER
"""

## Python Libraries
import torch
torch.manual_seed(123)
import numpy as np 
from torch import nn
import plotly
import plotly.graph_objects as go 
import plotly.figure_factory as ff
from csv import writer
import os
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

## Local files
from dataset import ModeChoiceDataset
from models import ANN


### ENVIRONMENT VARIABLES ###
OUTPUT_FOLDER = 'output'
EXPERIMENT_RESULTS = "Results.csv"

def save_plots(data, fpath)-> None:
    """
    Function to save plotly plots for later use. They are saved as JSON file 
    """
    if type(data) == go.Figure:
        plotly.io.write_json(data, fpath, pretty=True)
    else:
        raise TypeError("Save is not supported for {} datatype".format(type(data)))

def get_exp_number()->int:
    experiments = os.listdir(OUTPUT_FOLDER)
    if experiments:
        experiments = [int(e) for e in experiments]
        experiments.sort()
        return experiments[-1] + 1
    else:
        return 1

def create_folder(path)-> None:
    if not os.path.exists(path):
        os.mkdir(path)

def to_csv(data: list) -> None:
    """
    Function to save data to CSV. 
    """
    # Add Experiment to csv sheet
    with open(EXPERIMENT_RESULTS, 'a', newline='') as f_object: 
        # Pass the CSV  file object to the writer() function
        writer_object = writer(f_object, delimiter=";")
        # Result - a writer object
        # Pass the data in the list as an argument into the writerow() function
        writer_object.writerow(data)  
        # Close the file object
        f_object.close()
if not os.path.isfile(EXPERIMENT_RESULTS):  
    data_to_csv = [
        "EXP",
        "Hidden Layer",
        "Learning Rate",
        "Batch Size",
        "Epochs",
        "Optimizer",
        "Loss Func",
        "Train Loss",
        "Train Accuracy",
        "Test Accuracy",
        "Test F1-Score"]
    to_csv(data_to_csv)



def class_imbalance_plot(dataloader)->go.Figure:
    MODES = {1: "car",
                2: "pass",
                3: "bus", 
                4: "train",
                5: "walk",
                6: "bike"}
    bar = go.Figure()
    for itr, (data, target) in enumerate(dataloader):
        unique, counts = np.unique(target, return_counts=True)
        count_data = dict(zip(unique, counts))
        bar.add_trace(go.Bar(x=[MODES[idx] for idx in count_data.keys()],
                            y=list(count_data.values()),
                            name=itr))

    bar.update_xaxes(title_text='Available Modes')
    bar.update_yaxes(title_text='Count')
    bar.update_layout(title="Imbalance in Dataset", title_x=0.5)
    bar.update_layout(margin={"r":20,"t":50,"l":20,"b":20})
    return bar



############################# MODEL TRAINING, VALIDATION, TESTING #############################
def train(model, train_loader, loss_fn, optimizer): 
    # Variables to save the training loss and accuracy
    train_loss = 0.0
    train_correct = 0

    # Put model in training mode
    model.train()
    for X, y in train_loader:
        # Compute prediction and loss
        output = model(X)
        loss = loss_fn(output, y)

        # Back-propogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Training loss
        train_loss += loss.item() * X.size(0)
        # Accuracy
        scores, predictions = torch.max(output.data, 1)
        train_correct += (predictions == y).sum().item()

    train_loss = train_loss / len(train_loader.sampler)
    train_acc = train_correct / len(train_loader.sampler) * 100
    return train_loss, train_acc

def test(model, test_loader)->tuple((float, float, np.array)):
    model.eval()
    with torch.no_grad():    
        for idx, (X, y) in enumerate(test_loader):
            output = model(X)
            scores, y_hat = torch.max(output.data, 1)

            accuracy = accuracy_score(y, y_hat)
            f1_res = f1_score(y, y_hat, average='macro')
            z = confusion_matrix(y, y_hat, labels=list(testloader.dataset.labels.keys()))
    return accuracy, f1_res, z



if __name__ == "__main__":

    ###### GRID SEARCH PARAMETERS ######
    gridParams = {
        'hidden': [[3,5], [3,5,7]],
        'batch_size': [72, 120, 360],
        'lr_rate': [0.01,0.001, 0.0001, 0.00001],
        'epochs': [200],
        'optimizer': ['Adamax', 'Adagrad']
        }
    grid = ParameterGrid(gridParams)
    print("RUNNING {} EXPERIMENTS.... \n".format(len(grid)))
  
    for params in grid:
        ##### HYPER PARAMETERS ######
        hidden = params['hidden']           # Number of layers and hidden nodes in DNN
        learning_rate = params['lr_rate']   # Learning rate for optimizer
        batch_size = params['batch_size']   # Batch size for training and validation data
        epochs = params['epochs']           # Epcochs for training
        valid_size = 0.1                    # percentage split of the training set used for the validation set. Should be a float in the range [0, 1].
        shuffle = True                      # If to shuffle train and validation dataset before split
        verbose = True                      # FLAG for printing information. 
        ##############################

        #### LOADING DATA #####
        # Training Data and Validation Data ##
        path = "data/modeData.csv"
        train_data = ModeChoiceDataset(csv_path=path, data_type="train", normalize=True, verbose=False)  

        classes = train_data.y.unique()
        num_classes = [(train_data.y == clas).sum().item() for clas in classes]
        class_weights = [1 - (numcls/len(train_data)) for numcls in num_classes]
        samples_weight = np.array([class_weights[t.item()] for t in train_data.y])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weigth = samples_weight.double()
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=sampler)
        
        # Test Data ##
        test_data = ModeChoiceDataset(csv_path=path, data_type="test", normalize=True, verbose=False)  
        testloader = torch.utils.data.DataLoader(test_data, batch_size=400)

        ##### MODEL  #####
        in_dims = train_data.X.shape[1]
        out_dims = len(train_data.labels)
        model = ANN(input_dim=in_dims, output_dim=out_dims, hidden_dims=hidden)
        if verbose: print("\n", model)

        #### OPTIMIZER AND LOSS FUNC #####
        if params['optimizer'] == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if params['optimizer'] == "Adamax":
            optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
        if params['optimizer'] == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        if params['optimizer'] == "Adagrad":
            optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

        # Class Weighted Loss function## 
        class_weights = torch.FloatTensor(class_weights)
        loss_fn = nn.CrossEntropyLoss(weight = class_weights)

        ###################################################################################################################
        # Variable
        history = {'epoch'      :list(),
                   'train_loss' :list(),
                   'train_acc'  :list()}
       
        #### LOOP ######
        if verbose: 
            print("####################### HYPERPARAMETERS #########################")
            print('Hidden: {}\t Batch Size: {} \t LR: {} \t Epochs: {}'.format(
                    params['hidden'], params['batch_size'], params['lr_rate'], params['epochs']))
            print("#################################################################")
        ## Training  ###
        if verbose:
            print("\nEpoch \tTRN-Loss \tTRN-Acc %")
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train(model, trainloader, loss_fn, optimizer)

            # Bookeping
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            if verbose:
                print("{}/{} \t  {:.3f} \t  {:.3f}".format(epoch,epochs,train_loss,train_acc))
        
        ## Testing  ###
        test_acc, test_f1Score, zmat = test(model, testloader)
        if verbose:
            print("Finshing with Training for {} epochs".format(epochs))
            print("\tTEST-Acc: {:.3f}% \tTEST-F1: {:.3f}".format(test_acc*100, test_f1Score))
            print("\n")
        ##################################################################################################################


        # RESULTS ##
        # Plot for Loss
        lossPlot = go.Figure()
        lossPlot.add_trace(go.Scatter(x=history["epoch"],
                                      y=history["train_loss"],
                                      name="train loss"))
        
        lossPlot.update_xaxes(title_text='Epochs')
        lossPlot.update_yaxes(title_text='Loss')
        lossPlot.update_layout(title="Loss Plot", title_x=0.5)
        lossPlot.show()

        # Plot for accuracy
        accPlot = go.Figure()
        accPlot.add_trace(go.Scatter(x=history["epoch"],
                                      y=history["train_acc"],
                                      name="train accuracy"))
        
        accPlot.update_xaxes(title_text='Epochs')
        accPlot.update_yaxes(title_text='Accuracy %')
        accPlot.update_layout(title="Accuracy Plot", title_x=0.5)
        accPlot.show()

        # Plot for imbalance in data
        bar = go.Figure()
        for itr, (data, target) in enumerate(trainloader):
            unique, counts = np.unique(target, return_counts=True)
            count_data = dict(zip(unique, counts))
            bar.add_trace(go.Bar(x=[train_data.labels[idx] for idx in count_data.keys()],
                                y=list(count_data.values()),
                                name="Batch {}".format(itr)))

        bar.update_xaxes(title_text='Available Modes')
        bar.update_yaxes(title_text='Count')
        bar.update_layout(title="Imbalance in Dataset", title_x=0.5)
        bar.update_layout(margin={"r":20,"t":50,"l":20,"b":20})
        bar.show()

        # Plot for confusion matrix
        z_text = [[str(y) for y in x] for x in zmat]
        confusionMatrix = ff.create_annotated_heatmap(zmat, x=list(train_data.labels.values()), y=list(train_data.labels.values()), annotation_text=z_text)
        # add custom xaxis title
        confusionMatrix.add_annotation(dict(font=dict(color="black",size=14),
                                x=0.5,
                                y=-0.15,
                                showarrow=False,
                                text="Predicted value",
                                xref="paper",
                                yref="paper"))

        # add custom yaxis title
        confusionMatrix.add_annotation(dict(font=dict(color="black",size=14),
                                x=-0.15,
                                y=0.5,
                                showarrow=False,
                                text="Real value",
                                textangle=-90,
                                xref="paper",
                                yref="paper"))
        confusionMatrix.show()

        #### Saving Results ####
        # Experiment Directory
        exp_number = get_exp_number()
        path = os.path.join(OUTPUT_FOLDER, str(exp_number))
        create_folder(path)

        # Save plots
        save_plots(lossPlot, os.path.join(path, "loss_plots.pkl"))
        save_plots(accPlot, os.path.join(path, "acc_plots.pkl"))
        save_plots(bar, os.path.join(path, "data_imbalance.pkl"))
        save_plots(confusionMatrix, os.path.join(path, "confusionMatrix.pkl"))
        
        # Save data to csv
        data_to_csv = [
                exp_number,
                str(params['hidden']),
                params['lr_rate'],
                params['batch_size'],
                epochs,
                params['optimizer'],
                'Cross Entropy',
                train_loss, 
                train_acc,
                test_acc,
                test_f1Score]
        to_csv(data_to_csv)   