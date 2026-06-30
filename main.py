import torch
import torch.nn as nn
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.evaluation import *
from src.visualisation import *
from src.dataloader import *
from src.dataloader_indices import *
from src.config import *
from src.utils import *
from main_config import *
from src.models import *
from src.train import *

create_dirs()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

def run_test():
    DEVICE = get_device()

    print('Running Test Mode.')
    
    print('Loading Dataset...')
    

    if MODEL_FORM == 'RGB':
        train_loader, val_loader, test_loader = load_rgb_data()

        model = load_model(CHECKPOINT_PATH / 'rgb_model.pth', MODEL_FORM, pretrained=False)
    
    elif MODEL_FORM == 'MS':
        train_loader, val_loader, test_loader = load_ms_data()

        model = load_model(CHECKPOINT_PATH / 'ms_model.pth', MODEL_FORM, pretrained=False)

    elif MODEL_FORM == 'IND':
        train_loader, val_loader, test_loader = get_dataloader_indices()

        model = load_model(CHECKPOINT_PATH / 'ind_model.pth', MODEL_FORM, pretrained=False)

    test_model(model, test_loader, MODEL_FORM.lower())

def train_model():

    DEVICE = get_device()

    print('Loading Dataset...')
    
    if MODEL_FORM == 'RGB':

        # Loading Data
        train_loader, val_loader, test_loader = load_rgb_data()

        #Loading the Model
        model = resnet18_rgb_model()

        #Loss Function & Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr = 1e-4
        )

        print('Loss Function : ', criterion)
        print('Optimizer : ', optimizer)

        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.2, patience=5,  min_lr=1e-6
        )

        # Running the Training Loop
        history = training_loop(train_loader, val_loader, model, criterion, optimizer, 'rgb_model.pth', scheduler)

        # Plot all Findings
        plot_all_evaluation(history, model, test_loader, 'rgb')



    elif MODEL_FORM == 'MS':
        
        # Loading Data
        train_loader, val_loader, test_loader = load_ms_data()

        #Loading the Model
        model = resnet18_ms_model()

        #Loss Function & Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr = 1e-4
        )

        print('Loss Function : ', criterion)
        print('Optimizer : ', optimizer)

        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.2, patience=5,  min_lr=1e-6
        )

        # Running the Training Loop
        history = training_loop(train_loader, val_loader, model, criterion, optimizer, 'ms_model.pth', scheduler)

        # Plot all Findings
        plot_all_evaluation(history, model, test_loader, 'ms')



    elif MODEL_FORM == 'IND':

        # Loading Data
        train_loader, val_loader, test_loader = get_dataloader_indices()

        #Loading the Model
        model = resnet18_ind_model(in_channels=8, pretrained=True)

        model_ind = model.to(DEVICE)

        print('Conv Layer : ', model_ind.conv1)
        print('Final Layer : ', model_ind.fc)


        #Loss Function & Optimizer
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model_ind.parameters(),
            lr=1e-4
        )

        print('Loss Function : ', criterion)
        print('Optimizer : ', optimizer)

        history = training_loop(train_loader, val_loader, model, criterion, optimizer, 'ind_model.pth')

        plot_all_evaluation(history, model, test_loader, 'ind')



def main():
    print('\n EuroSAT Model Pipeline.')

    if RUN_TYPE == 'Test':
        run_test()

    elif RUN_TYPE == 'Train':
        train_model()

    else:
        print('Enter Valid Mode.')

if __name__ == '__main__':
    main()


