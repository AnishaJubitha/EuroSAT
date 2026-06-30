# EuroSAT RGB and Multispectral Classification

This project compares land-use classification performance on the EuroSAT dataset using RGB imagery, 13-band Sentinel-2 multispectral imagery, and RGB with engineered spectral indices. ResNet18 is used as the main classification model, with results evaluated using accuracy, macro-F1, classification reports and confusion matrices.


### Models

The project supports three model types:
- RGB	: RGB ResNet18 baseline
- MS	: 13-band multispectral ResNet18
- IND	: RGB + spectral indices ResNet18


### Environment Setup

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

### Configurations and Executions

1. Execution is controlled from main_config.py. 

2. Update the following variables before running main.py:

- RUN_TYPE - can be 'Train' or 'Test' depending upon what you want to do.
- MODEL_FORM - can be 'RGB', 'MS' or 'IND' 

The MODEL_FORM helps to train or test the model you choose. 

3. From the project root folder, run: 

python main.py

4. Your Final Model will be Stored inside -

- 'Results/Checkpoint'

the model names are 'rgb_model.pth', 'ms_model.pth' or 'ind_model.pth'.

Any other models saved are previous versions trained before submission.



### EDA Results

The EDA Results can be tested using the below notebooks -
1. 01_dataset_validation.ipynb
2. 02_preprocessing.ipynb


### RESULTS

All models are stored inside the 'results' folder.
The folder contains 

'checkponts' - final trained models 
'metrics' - plots after evaluation
'eda' - EDA plots and results