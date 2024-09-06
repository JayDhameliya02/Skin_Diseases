# Skin Lesion Classifier

This project is a Flask web application that allows users to upload an image of a skin lesion along with metadata such as age, sex, and localization. The application uses a pre-trained convolutional neural network (CNN) to predict the type of skin disease.

## Project Structure

- `app/`: Contains the Flask application.
- `data/`: Contains the dataset files.
- `models/`: Contains the trained model file.
- `src/`: Contains scripts for data preprocessing, model training, evaluation, visualization, and test data generation.
- `test_data/`: Contains generated test images and metadata.
- `notebooks/`: Contains Jupyter notebooks for exploratory data analysis.
- `run_project.py`: Script to run the entire project pipeline.
- `requirements.txt`: List of project dependencies.
- `README.md`: Project documentation.

## Installation

Python Downloads, Conda Downloads, Git Downloads

## The instructions below will help you set up the environment.

Prerequisites To use the program, ensure that Python v3 and the following libraries are installed on your operating system:

1. Flask
2. tensorflow
3. numpy
4. pandas
5. Pillow
6. scikit-learn
7. imblearn
8. matplotlib
9. seaborn

You can develop in Jupyter Notebooks as well as editors like VS Code or PyCharm.

## Follow below step for build Environment

Installation on Windows using terminal:
1. conda create --name myenv python=3.8
2. conda activate myenv
3. pip install -r requirements.txt
4. python app.py

## Local host path
Copy ('http://127.0.0.1:5000/') your local host path and insert it into any browser to show our dashboard.

NOTE : "Edit Model path in 'app.py' file. "
  
