import tensorflow as tf
from sklearn.metrics import accuracy_score
import numpy as np

def main():
    # Load the trained model
    model = tf.keras.models.load_model('wildlife_spotting_model.h5')

    # Load your test dataset
    # Replace X_test and y_test with your test dataset
    # Ensure that the test dataset is preprocessed similarly to the training dataset
    # Normalize pixel values to the range [0, 1] if necessary
    y_pred = model.predict(X_test)

    # Convert probabilities to binary predictions
    y_pred_binary = np.round(y_pred).astype(int)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred_binary)
    print('Test accuracy:', accuracy)

if __name__ == "__main__":
    main()
