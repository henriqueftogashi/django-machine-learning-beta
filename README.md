# django-machine-learning-beta

This Django app (beta) accepts any csv file to run a classification model model among Logistic Regression, Random Forest (10 interations) and xgboost. The user has an option to profiling and preprocess data; addressing missing values, one hot encoding and feature scaling. The 3 models can be selected simultaneously for accuracy comparison. The results and validation are provided as confusion matrix and ROC curve.

This is my first Django project and is still a early beta version. I hope the code published here
can help people that are also learning Django.

## Getting Started

Install atom
packages:
platformio-ide-terminal. After downloading go to settings and change "Shell Override" to C:\WINDOWS\system32\cmd.exe
atom-django
autocomplete-html-entities
autocomplete-python

Installing anaconda and python:
https://conda.io/docs/user-guide/install/windows.html

conda info --envs #list environments

conda create --name myDjangoEnv python=3.6 #creates environment #create one environment for django project because if libraries get updated it can break the app

activate MyDjangoEnv #activate environment

conda install django #installs django in the environment 

pip install pandas
pip install plotly
pip install sklearn
pip install scipy
pip install xgboost


## Database

The project is set up for a postgres database. You need to change the DB details such as owner and password.

For SQLlite replace in setting.py

```
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}
```
