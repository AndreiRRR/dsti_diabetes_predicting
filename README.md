# Predicting Diabetes Outcome for Women (Flask Web Application)

The project is a part of the 'Machine Learning with Python Labs' course, DSTI, 2025.

## Description

The goal of the project is to build and deploy a predictive model capable of classifying individuals as diabetic or non-diabetic. 
The end product is a web application that allows users to input individual health metrics and receive real-time diabetes predictions from the trained machine learning model.

## Repository structure

The repository includes the application codes as well the data file and the report document and files. 
The directories in the repository are organized according project documentation. 

Directories are :
1_JupitersNotebooks ==> directory contains Jupyter Notebooks that includes DEA, feature engineering and selection, model training, comparison and evaluation. 
2_WebApp            ==> directory incorporates the web application files.
3_Report            ==> directory includes the project report.
4_WebApp_Video      ==> directory contains the small video of the web application running. 
Data                ==> directory serves to store project data set


## Web Application Prerequisites

    Windows operating system
    Python 3.7 or higher
    PyCharm IDE (Community or Professional edition)

## Web Application Installation and Setup
1. Installing PyCharm on Windows

    Download PyCharm from the official website: https://www.jetbrains.com/pycharm/download/
    Run the downloaded installer (.exe file)
    Follow the installation wizard: Click "Install"    
    Once installation completes, click "Finish" and launch PyCharm


2. Creating a New Flask Project in PyCharm

    Open PyCharm

    Click "New Project" on the welcome screen

    In the "New Project" dialog:
        Choose a location for your project
        Select "Flask" as the project type (if available, otherwise select "Pure Python")
        Choose your Python interpreter 

    Click "Create"

    If Flask wasn't available as a project type:
        Create a regular Python project
        Then install Flask using PyCharm's terminal: pip install flask

3. Downloading and Adding Project Files

    Clone this repository or download the source files:
    git clone https://github.com/AndreiRRR/dsti_diabetes_predicting.git 


    In PyCharm:
        Go to File → Open
        Navigate to the project folder you just cloned/downloaded
        Select the folder and click "OK"

    The project should contain the following structure (download files from /dsti_diabetes_predicting/tree/main/2_WebApp directory):
    
    predicting-diabetes-outcome-flask/
    ├── templates/
    │   └── index.html
    ├── Diabetes_Model.pkl
    ├── main.py
    └── requirements.txt


4. Installation of Dependencies

    Open the terminal in PyCharm (Alt+F12)
    Install the required packages using:
    pip install -r requirements.txt


5. Running the Application

    Ensure you have all dependencies installed
    Run the Flask application:
    python .\main.py

    Open your web browser and navigate to:
    http://127.0.0.1:5000

    Use the web form to input patient data and get diabetes predictions.

6. Application Files Overview


-- requirements.txt 
The list of all the Python libraries and dependencies required to run the project. 

-- main.py
The main Flask application file that:
    Loads the trained model (Diabetes_Model.pkl)
    Defines routes for the web application
    Handles form submissions and makes predictions

-- Diabetes_Model.pkl
The pre-trained machine learning model (Pickle format) used for predictions.

-- templates/index.html
The HTML template that provides:
    A form for user input
    Displays prediction results
    Basic interface styling



## Authors

Mahamadou Ousmane KEITA	
Andrei Roumiantsev
TEFOPA JIONGO Rosine	
Malathi Karuppasamy	
Jean Daniel HOUSSOU	
Arthur Choisnet