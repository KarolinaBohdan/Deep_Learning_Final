# Green Patent Detection Project

## Overview
This project is centered around the detection of green patents using various machine learning techniques. It aims to harness patent data to identify environmentally friendly innovation.

## Assignments
1. **Data Collection**: Gathering of patent data from multiple sources.
2. **Data Cleaning**: Processing the collected data to remove duplicates and irrelevant entries.
3. **Feature Extraction**: Identifying key features that influence the categorization of patents as green or non-green.
4. **Model Development**: Applying various machine learning models to the extracted features to classify patents.
5. **Evaluation**: Analyzing model performance metrics to validate the model effectiveness.

## Methods
- **Data Mining**: Used for extracting relevant patent data.
- **Natural Language Processing (NLP)**: Techniques applied for text processing and feature extraction from patent descriptions.
- **Machine Learning Algorithms**: Multiple algorithms applied including Decision Trees, Random Forests, and Support Vector Machines (SVM).

## File Structure
```
Deep_Learning_Final/
├── data/
│   ├── raw_data/
│   ├── cleaned_data/
├── notebooks/
│   ├── exploratory_data_analysis.ipynb 
│   ├── model_development.ipynb
├── src/
│   ├── data_processing.py
│   ├── model_training.py
│   ├── evaluation.py
├── results/
│   ├── model_performance_metrics.csv
│   ├── figures/
├── README.md
```  

## Results
The models developed achieved a highest accuracy of 85% on the test dataset. Detailed performance metrics such as precision, recall, F1-score, and ROC-AUC are provided in `results/model_performance_metrics.csv`.

## Usage Instructions
1. Clone the repository using:
   ```bash
   git clone https://github.com/KarolinaBohdan/Deep_Learning_Final.git
   ```
2. Navigate into the project folder:
   ```bash
   cd Deep_Learning_Final
   ```
3. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the data processing script:
   ```bash
   python src/data_processing.py
   ```
5. Train models using notebooks located in the `notebooks/` directory.