# GITGAN

## Installation

1. **Clone the Repository:**  
```bash
git clone git@github.com:Kang1121/GITGAN.git
cd GITGAN
```

2. **Set Up the Environment:**  
```bash
conda env create --file environment.yaml
conda activate gitgan
```


## Data Acquisition & Setup

1. **Download Datasets:**  
Fetch the required datasets from [this link](https://works.do/5MZGWeN).

2. **Place in Code Directory:**  
After downloading, ensure the datasets are placed directly within the `GITGAN` code directory.

## Configuration & Usage

**Customizing Settings:**  
Dive into the **configs** directory to find and modify hyperparameters and other settings to cater to your requirements.

**Running the Model:**

- **With Distributed Data Parallel (DDP):**  
```bash
torchrun --nproc_per_node=NUM_GPUs main.py --config_file configs/DATASET2RUN.yaml
```
Replace `NUM_GPUs` with the number of GPUs you wish to use.

- **Without DDP:**  
```bash
python main.py --config_file configs/DATASET2RUN.yaml
```
