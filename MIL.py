import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import  StandardScaler
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import numpy as np
from matplotlib import pyplot as plt
import argparse
import warnings


### Creating the model for the MIL method

class Classif_MIL(nn.Module):
    def __init__(self,num_features = 279, num_classes=1, dropout_rate=0.5):
        super(Classif_MIL, self).__init__()
       

        self.fc_layers = nn.Sequential(
            nn.Linear(num_features , 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate/1.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate/2),
            nn.Linear(128, num_classes)
        )


    def forward(self, slices):
        out = self.fc_layers(slices)
        return out

    
    def configure_optimizers(self, lr=1e-4):
        # weight decay ==> L2 regularization
        return optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
    

#### Creating the Dataloader adapted to the MIL method
class MIL_Slice_Dataset(Dataset):
    
    def __init__(self, dataframe_slices, labels_df ):
        # Preprocessing the data and initializing necessary variables
        # Here, StandardScaler is used to standardize the data
        patient_nums = dataframe_slices['patient_num']
        dataframe_slices_without_patient_num = dataframe_slices.drop(columns=['patient_num'])

        scaler = StandardScaler()
        normalized_df = scaler.fit_transform(dataframe_slices_without_patient_num)
        normalized_df = pd.DataFrame(normalized_df, columns=dataframe_slices_without_patient_num.columns)

        self.normalized_slices = pd.DataFrame(normalized_df, columns=dataframe_slices.columns)
        self.normalized_slices['patient_num'] = patient_nums
        self.normalized_slices = self.normalized_slices[dataframe_slices.columns]

        self.corresp = {'CCK': 0, 'CHC':1}  # Corresponding class labels
        self.labels_df = labels_df

        
    
    def __len__(self):
        return len(self.labels_df['patient_num'].unique())

    def __getitem__(self, idx):
        # Getting data for a specific patient
        patient_id = self.normalized_slices['patient_num'].unique()[idx]
        label_patient =  self.get_label_tensor(patient_id)  
        patient_data_slices =  self.normalized_slices[self.normalized_slices['patient_num'] == patient_id]

        slices = []

        for _, row in patient_data_slices.iterrows(): # one unit of the batch = one patient 
    
            slices.append(torch.tensor(list(row.drop('patient_num'))))


        return torch.tensor(patient_id), torch.stack(slices), label_patient, torch.tensor(len(slices))

    def get_label_tensor(self, patient_id):
        # Getting label tensor for a patient
        filtered_labels = self.labels_df[self.labels_df['patient_num'] == patient_id]['classe_name']
        if not filtered_labels.empty:
            first_classe_name = filtered_labels.iloc[0]
            if first_classe_name in self.corresp:
                label_tensor = torch.tensor(self.corresp[first_classe_name], dtype=torch.float32)
                return label_tensor
            else:
                # Handle case where first_classe_name is not in corresp
                return torch.tensor(0, dtype=torch.float32)  # Example default value
        else:
            # Handle case where filtered_labels is empty
            return torch.tensor(0, dtype=torch.float32)  # Example default value

def patient_collate_fn(batch):
    """
    Custom collate function to handle batches of patients
    """
    patient_ids, slices, labels,n_slices = zip(*batch)
    slices_flattened = [torch.tensor(slic) for patient_slices in slices for slic in patient_slices]

    labels = [torch.tensor(label) for label in labels]
    labels = torch.stack(labels) 

    return patient_ids, torch.stack(slices_flattened), labels,n_slices


class LymphoDataLoader:
    """
    Data Loader calling our previous MIL_Slice_Dataset to be processed by the DL model.
    """
    def __init__(self, dataframe_slices, label_df, batch_size, shuffle=True, num_workers=0):
        # Initializing the DataLoader with our MIL_Slice_Dataset
        self.dataset = MIL_Slice_Dataset(dataframe_slices, label_df)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def get_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=patient_collate_fn
        )
    

## Training and validation functions

def custom_train_epoch(model, train_loader, optimizer, criterion, device, topk):
    """
    Customed function used to train the model for one epoch, processing the batches of patients 
    and aggregating them following the Top K% procedure to compute the loss and backpropagate.
    """
    
    model.train()
    loss_total = 0
    num_batches = 0

    for batch in train_loader: # 1 batch = at least 1 patient with all of its instances = slices
        
        patient_ids, slices, labels, n_slices = batch
        slices = slices.to(device)
        labels = labels.to(device).float()
        
        optimizer.zero_grad()
        outputs = model(slices)
        aggregated_outputs = torch.tensor([], device=device)
        
        deb = 0 # dummy variable to know where the patients' slices start and end
        
        for fin in n_slices :
            
            patient_outputs = outputs[deb:deb+fin]
            deb += fin

            aggregated_output = aggregate_patient_predictions(patient_outputs, topk, aggregation='mean')
            aggregated_outputs = torch.cat((aggregated_outputs, aggregated_output), dim=0)
                    
        loss = criterion(aggregated_outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()

        loss_total += loss.item()
        num_batches += 1

    return loss_total / num_batches


def custom_validate_epoch(model, val_loader, device, topk):
    """
    Customed function used to compute the loss and metrics on the validation set, 
    processing the batches of patients and aggregating them following the Top K% procedure
    """

    model.eval()
    num_batches = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in val_loader:
            patient_ids, slices, labels, n_slices = batch

            slices = slices.to(slices)
            labels = labels.to(device).float()



            outputs = model(slices)
            outputs = torch.sigmoid(outputs)

            aggregated_outputs = torch.tensor([], device=device)
            deb = 0

            for fin in n_slices:
                patient_outputs = outputs[deb:deb+fin]
                deb += fin

                aggregated_output = aggregate_patient_predictions(patient_outputs, topk, aggregation='mean')
                aggregated_outputs = torch.cat((aggregated_outputs, aggregated_output), dim=0)

            num_batches += 1

            #Convert aggregated outputs to binary predictions
            predictions = torch.round(aggregated_outputs).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions)

    #Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    metrics = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    return metrics

def aggregate_patient_predictions(predictions_tensor, topk=15, aggregation='mean'):
    """
    Aggregates predictions for a single patient based on the topK% method.
    """
    num_instances = predictions_tensor.size(0)
    topk = max(1, int((topk / 100.0) * num_instances))
    if aggregation == 'mean':
        topk_values, _ = torch.topk(predictions_tensor, k=topk, largest=True, dim=0)
        aggregated_prediction = topk_values.mean(dim=0)
    elif aggregation == 'max':
        aggregated_prediction = torch.max(predictions_tensor, dim=0)[0]
    else:
        raise ValueError("Unsupported aggregation method. Choose 'mean' or 'max'.")

    return aggregated_prediction.unsqueeze(0) 



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    # Adding arguments
    parser.add_argument('--Val', help='If you cant to look yourself for the best K parameter value', required=True)
    parser.add_argument('--Test', help='If you want to test the model', required=True)
    parser.add_argument('--n_runs_val', help='Number of independent rund fir Validation', required=False)
    parser.add_argument('--n_runs_test', help='Number of independent rund for Validation', required=False)
    
    args = parser.parse_args()
    
    # Accessing arguments
    Val = str(args.Val)
    Test = str(args.Test)
    
    if Val=='True':
        Val = True
    else:
        Val = False

    if Test=='True':
        Test = True
    else:
        Test = False

    if Val:
        n_runs_val = int(args.n_runs_val)
        print('Validation for the MIL method (finding the best K value)')
        K_values = [15,20,25,30,50, 75,90,100]
        Bal_acc = np.zeros((n_runs_val, len(K_values)))

        for i in tqdm(range(len(K_values)), leave = False):
            topK  = K_values[i]
            BAccs = []
            precisions = []
            recalls = []
            f1s = []
            for _ in tqdm(range(n_runs_val)):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                #print(f'Device : {device}')
                model = Classif_MIL().to(device)
                optimizer = model.configure_optimizers()
                scheduler = StepLR(optimizer, step_size=5, gamma=0.6) # adaptative lr to better converge to maxima
                pos_weight = torch.tensor([1], device=device)
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                n_epochs = 5


                ### We load the train and val dataset created earlier : 

                train_dataset = pd.read_csv('data/slices_train.csv')
                label_train = pd.read_csv('data/labels_train.csv')

                val_dataset = pd.read_csv('data/slices_val.csv')
                label_val = pd.read_csv('data/labels_val.csv')

                test_dataset = pd.read_csv('data/slices_val.csv')
                label_test = pd.read_csv('data/labels_val.csv')
              

                max_BA = 0
                best_metrics = {}
                for epoch in range(n_epochs):
                    
                    warnings.filterwarnings("ignore", message="To copy construct from a tensor.*")
                    data_module = LymphoDataLoader(train_dataset, label_train, batch_size=4, shuffle=True) 
                    train_loader = data_module.get_dataloader()
                    train_loss = custom_train_epoch(model, train_loader, optimizer, criterion, device, topk = topK)
                    scheduler.step()
                    
                    
                    data_module_val = LymphoDataLoader(val_dataset, label_val, batch_size=4, shuffle=True)
                    val_loader = data_module_val.get_dataloader()
                    metrics = custom_validate_epoch(model, val_loader, device, topk = topK)
                    if metrics['balanced_accuracy']>max_BA:
                        max_BA = metrics['balanced_accuracy']
                        best_metrics = metrics

                    
                BAccs.append(max_BA)
            Bal_acc[:,i] = BAccs

        mean_BA = np.mean(Bal_acc, axis = 0)
        std_BA = np.std(Bal_acc, axis = 0)

        plt.plot(K_values, mean_BA, label = 'Balanced Accuracy', color = 'Blue')
        plt.fill_between(K_values,mean_BA-std_BA, mean_BA+std_BA, alpha=1, edgecolor='#3F7F4C', facecolor='#7EFF99',linewidth=0)
        plt.xlabel('Top K values')
        plt.ylabel('Balanced Accuracy')
        plt.title('Balanced accuracy for MIL model')
        plt.show()

    if Test:
        n_runs_test = int(args.n_runs_test)
        print('Testing the MIL method')
        K_values = [100]
        Bal_acc = np.zeros((n_runs_test, len(K_values)))
        precision = np.zeros((n_runs_test, len(K_values)))
        recall  = np.zeros((n_runs_test, len(K_values)))
        F1_score = np.zeros((n_runs_test, len(K_values)))


        for i in tqdm(range(len(K_values)), leave = False):
            topK  = K_values[i]
            BAccs = []
            precisions = []
            recalls = []
            f1s = []
            for _ in tqdm(range(n_runs_test)):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = Classif_MIL().to(device)
                best_model = None
                optimizer = model.configure_optimizers()
                scheduler = StepLR(optimizer, step_size=5, gamma=0.6) # adaptative lr to better converge to maxima
                pos_weight = torch.tensor([1], device=device)
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                n_epochs = 5


                ### We load the train and val dataset created earlier : 

                train_dataset = pd.read_csv('data/slices_train.csv')
                label_train = pd.read_csv('data/labels_train.csv')

                val_dataset = pd.read_csv('data/slices_val.csv')
                label_val = pd.read_csv('data/labels_val.csv')

                test_dataset = pd.read_csv('data/slices_val.csv')
                label_test = pd.read_csv('data/labels_val.csv')

                max_BA = 0
                res_BA = 0
                res_precision = 0
                res_recall = 0
                res_F1 = 0
                best_metrics = {}
                for epoch in range(n_epochs):
                    
                    #print(f'Epoch {epoch}/{n_epochs}')
                    warnings.filterwarnings("ignore", message="To copy construct from a tensor.*")
                    data_module = LymphoDataLoader(train_dataset, label_train, batch_size=4, shuffle=True) 
                    train_loader = data_module.get_dataloader()
                    train_loss = custom_train_epoch(model, train_loader, optimizer, criterion, device, topk = topK)
                    scheduler.step()

                    data_module_val = LymphoDataLoader(val_dataset, label_val, batch_size=4, shuffle=True)
                    val_loader = data_module_val.get_dataloader()
                    metrics = custom_validate_epoch(model, val_loader, device, topk = topK)
                    if metrics['balanced_accuracy']>max_BA:
                        max_BA = metrics['balanced_accuracy']
                        best_model = model.state_dict().copy()
                

                data_module_test = LymphoDataLoader(test_dataset, label_test, batch_size=4, shuffle=True)
                model.load_state_dict(best_model)
                test_loader = data_module_test.get_dataloader()
                metrics = custom_validate_epoch(model, test_loader, device, topk = topK)
                
                res_BA = metrics['balanced_accuracy']
                res_precision = metrics['precision']
                res_recall = metrics['recall']
                res_F1 =  metrics['f1_score']
                
                BAccs.append(res_BA)
                precisions.append(res_precision)
                recalls.append(res_recall)
                f1s.append(res_F1)
            Bal_acc[:,i] = BAccs
            precision[:,i] = precisions
            recall[:,i] = recalls
            F1_score[:,i] = f1s

        mean_BA = np.mean(Bal_acc, axis = 0)
        std_BA = np.std(Bal_acc, axis = 0)

        mean_precision = np.mean(precision, axis = 0)
        std_precision = np.std(precision, axis = 0)

        mean_recall = np.mean(recall, axis = 0)
        std_recall = np.std(recall, axis = 0)

        mean_f1 = np.mean(F1_score, axis = 0)
        std_f1 = np.std(F1_score, axis = 0)
        print(f'Balanced accuracy on the test dataset : {mean_BA[0]} += {std_BA[0]}')
        print(f'Precision on the test dataset : {mean_precision[0]} += {std_precision[0]}')
        print(f'Recall on the test dataset : {mean_recall[0]} += {std_recall[0]}')
        print(f'F1 score on the test dataset : {mean_f1[0]} += {std_f1[0]}')