# Energy Efficiency
# Project Description

This study looked into assessing the heating load and cooling load requirements of buildings (that is, energy efficiency) as a function of building parameters. The buildings differ with respect to the glazing area, the glazing area distribution, and the orientation, amongst other parameters.

The effect of eight input variables (relative compactness, surface area, wall area, roof
area, overall height, orientation, glazing area, glazing area distribution) on two output
variables, namely heating load (HL) and cooling load (CL), of residential buildings is
investigated using a statistical machine learning framework.

The data must be provided in 10 columns for training. The description of the columns are stated below:
- “X1” (input) - Relative Compactness
- “X2” (input) - Surface Area
- “X3” (input) - Wall Area
- “X4”(input)  - Roof Area
- “X5” (input) - Overall Height
- “X6” (input) - Orientation
- “X7” (input) - Glazing Area
- “X8” (input) - Glazing Area Distribution
- “Y1” (output) - Heating Load
- “Y2” (output)- Cooling Load
The terms in double quotes are the column names.
The Web Application which implements this repository also provides the facility to retrain the model used to predict the heating and cooling load.
This repository implements "Energy Efficiency" project work done through internship under ineuron.ai.
With the help of this project we can estimate the heating and cooling loads required for a building for efficient energy consmption.



## Authors

- Subhransu Sekhar Mallick
- Itishree Samal

## Dataset
The dataset used for training the model is available here https://archive.ics.uci.edu/dataset/242/energy+efficiency
## Installation
### Requirements 
- Python 3.7+
- scikit-learn 1.0.2
- Random Forest
- pandas 1.3.5
- numpy 1.21.6
- streamlit
- matplotlib

### Setup
Install Machine Learning Libraries

```bash
pip install scikit-learn==1.0.2 Random Forest pandas==1.3.5
```
Install Library for hosting Web Application 
```bash
pip install streamlit
```
Install supporting libraries
```bash
pip install pandas==1.0.0 numpy==1.21.6
```
## Screenshots

![App Screenshot](https://drive.google.com/file/d/1q66HMmmxNEYSDod1XW0fFvQSocQ1QUrR/view?usp=sharing)


## Demo

- A working implementation of the project as a Web Application in this repository is available https://energy-efficient-building-load-calculation-app-3ukurqwqpujw46k.streamlit.app/

