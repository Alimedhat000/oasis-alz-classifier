# Pytorch Project Template


## Project Organization

```
├── .env               
├── .gitignore         
├── README.md          
├── data               <- (Not Version Controlled)
│   ├── processed      <- The final, processed data sets for modeling. 
│   └── raw            <- The original, data dump.
│
├── experiments        <- Experiment logs and results
│   └── .gitkeep
│
├── models             <- Trained and serialized models
│   └── .gitkeep
│
├── notebooks          <- Jupyter notebooks
│   └── .gitkeep
│
├── references         <- Data dictionaries, papers, manuals, and all other explanatory materials
│   └── .gitkeep
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── .gitkeep
│   └── figures        <- Generated graphics and figures to be used in reporting
│
└── src                <- Source code for use in this project.
    ├── __init__.py    <- Makes src a Python module
    ├── config.py      <- Store useful variables and configuration
    ├── data/          <- Data handling components
    │   ├── dataloader.py     <- PyTorch DataLoader implementations
    │   ├── dataset.py        <- Dataset classes
    │   ├── preprocessing.py  <- Data preprocessing functions
    │   └── transforms.py     <- Data transformation utilities
    │  
    ├── inference/     <- Model inference and evaluation components
    │   ├── evaluate.py       <- Evaluation scripts
    │   └── inference.py      <- Inference utilities
    │
    ├── model/         <- Model architecture 
    │   ├── __init__.py
    │   └── architecture.py   <- Model architecture definitions
    │   
    ├── training/      <- Training-related components
    │   ├── losses.py         <- Loss function definitions
    │   ├── metrics.py        <- Evaluation metrics
    │   ├── optimizer.py      <- Optimizer configurations
    │   ├── train.py          <- Training script
    │   └── trainer.py        <- Training loop implementation
    │ 
    └── utils/         <- Utility functions
        ├── device.py         <- Device management
        ├── logger.py         <- Logging utilities
        ├── plots.py          <- Visualization functions
        └── seed.py           <- Random seed utilities
```

--------

