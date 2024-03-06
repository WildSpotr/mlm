# WildSpotr CNN Model

This repository contains a simple Convolutional Neural Network (CNN) model trained for a wildlife spotting application. 
The model is implemented using TensorFlow/Keras and is designed to classify images as either containing wildlife or not.

## Usage

1. **Clone the Repository:**
   ```
   git clone https://github.com/WildSpotr/mlm.git
   cd mlm
   ```

2. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Prepare Your Dataset:**
   - Replace `data` and `labels` in the code with your dataset and corresponding labels.
   - Ensure that your dataset is structured appropriately and contains images of wildlife and non-wildlife scenes.

4. **Train the Model:**
   ```
   python train_model.py
   ```

5. **Evaluate the Model:**
   ```
   python evaluate_model.py
   ```

## Model Architecture

The CNN model architecture used in this project consists of several convolutional layers followed by max-pooling layers and dense layers. The final layer uses a sigmoid activation function for binary classification.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the GPL-3.0 License. See the [LICENSE](https://github.com/WildSpotr/mlm/blob/main/LICENSE) file for details.
