## Downloading Datasets

Link to the project report - [PDF](https://github.com/VrutikShah/SRFBN/blob/master/SuperResolutionUsingDL.pdf)

Div2K dataset - [Link](https://drive.google.com/drive/folders/1Rqb5Poe5oe2R7vFJhk9jw_ksxClYufsZ?usp=sharing)  
To download Set5 Dataset, run 
```
wget https://data.deepai.org/set5.zip
```  

## Installing Libraries  
For PyTorch please visit this [link](https://pytorch.org/) to install proper version of PyTorch. After that, run the following command to install all the required libraries
```
pip install -r requirements.txt
```


## Preparing Dataset and Augmenting Data
To prepare the dataset, change the parameters at the top of the file and run 
```
python3 prepare_data.py
```


## Training

For Training, several modes are available, please have a look at the top of the train.py file to decide some of the parameters to train the model with. Once the parameters are finalized, run 
```
python3 train.py
```


## Testing  
For testing, first finalize the parameters given at the begining of the test_model.py file. Once finalized run 
```
python3 test_model.py
```
