# Green Patent Detection

## Description
This project focuses on using deep learning techniques to detect green patents, which are patents that are related to environmentally friendly technologies. The goal is to build a model that can accurately classify these patents based on textual descriptions.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/KarolinaBohdan/Deep_Learning_Final.git
   cd Deep_Learning_Final
   ```
2. Set up a Python environment (recommended to use virtualenv or conda):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
The dataset used for this project is obtained from [insert dataset source]. It contains [brief description of the contents]. Make sure to download and place the dataset in the `data/` directory.

## Model Architecture
The model architecture consists of [brief description of the architecture, e.g., number of layers, types of layers, etc.]. The model is built using [framework used, e.g., TensorFlow, PyTorch].

## Training
To train the model, run the following command:
```bash
python train.py --epochs 50 --batch_size 32
```
Make sure to adjust the parameters according to your compute resources.

## Evaluation
After training the model, you can evaluate its performance with:
```bash
python evaluate.py --model_path path/to/saved_model
```
This will output accuracy, precision, recall, and F1 score metrics.

## Usage
To use the model for prediction on new data, run:
```bash
python predict.py --input_file path/to/input_data
```

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request. Provide a clear description of your changes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.