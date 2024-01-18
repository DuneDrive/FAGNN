# Feature Attention Graph Neural Network (FAGNN)

### Feature attention graph neural network for estimating brain age and identifying important neural connections in mouse models of genetic risk for Alzheimer’s disease

Hae Sol Moon, Ali Mahzarnia, Jacques Stout, Robert J Anderson, Cristian T. Badea, Alexandra Badea

[Link to preprint (doi: 10.1101/2023.12.13.571574)](https://doi.org/10.1101/2023.12.13.571574)

<img width="800" alt="FAGNN_schematics_v4" src="https://github.com/DuneDrive/FAGNN/assets/70248584/4cf35f49-37ad-4451-8216-f4ab4c5bab7a">


## Abstract
An implementation of FAGNN using diffusion MRI-based structural connectomes, biological traits, and behavioral metrics for brain age prediction.


## Usage
### Setup
Clone this repository to your local machine and install packages:
```
pip install -r requirements.txt
```

### Data Download
Download data used in the paper at:
[https://zenodo.org/records/10372075](https://zenodo.org/records/10372075).
Alternatively, use your customized data.

### Training
For customization, modify `net/model_FAGNN.py` based on your data. Modify line 31 of the code to define your data directory.
Then run this code to train the model:

```
python train_FAGNN.py
```

### Additional Information
For queries or issues, please contact: `<haesol.moon@duke.edu>`
