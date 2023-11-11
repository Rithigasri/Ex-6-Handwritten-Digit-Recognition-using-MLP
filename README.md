# SKILL ASSESSMENT:HANDWRITTEN DIGIT RECOGNITION USING MLP
## AIM:
To Recognize the Handwritten Digits using Multilayer perceptron.
##  EQUIPMENTS REQUIRED:
* Hardware – PCs
* Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook
## THEORY:
### 1. **Multilayer Perceptron (MLP)**
- **Architecture**: MLP is a feedforward neural network composed of multiple layers - an input layer, one or more hidden layers, and an output layer.
- **Fully Connected Layers**: Each neuron in a layer is connected to every neuron in the subsequent layer.
- **Activation Functions**: Typically, nonlinear functions (e.g., ReLU, Sigmoid, Tanh) are used in the hidden layers to introduce nonlinearity, allowing the network to learn complex patterns.

### 2. **Digit Recognition using MLP**
- **Data Preprocessing**: For digit recognition, datasets like MNIST (or similar) are common. Images are preprocessed by:
    - Normalizing pixel values to a range (commonly 0 to 1) for faster convergence.
    - Flattening the 2D images into 1D arrays to feed them into the network.

### 3. **Model Training and Components**

#### a. **Layers**
- **Input Layer**: Neurons represent input features (flattened pixel values).
- **Hidden Layers**: Multiple layers where features are transformed through weighted connections and activation functions.
- **Output Layer**: Neurons represent the possible classes (0-9 for digit recognition) with a probability distribution using Softmax activation.

#### b. **Loss Function and Optimizer**
- **Loss Function**: Categorical Cross-Entropy is often used for multiclass classification problems like digit recognition.
- **Optimizer**: Algorithms like Adam, SGD, etc., update network weights based on the computed loss.

#### c. **Training and Evaluation**
- **Forward Propagation**: Input data moves forward through the network; predictions are made.
- **Backward Propagation**: Based on the loss, gradients are calculated and weights are updated using optimization algorithms.
- **Validation and Testing**: Model performance is measured using a separate validation or test dataset.

### 4. **Visualization and Interpretation**
- **Training History**: Graphical representation of accuracy and loss over epochs helps understand model performance and potential overfitting.
- **Prediction Visualization**: Showing images with their predicted and actual labels provides a practical understanding of the model's accuracy.

### 5. **Interactive Prediction**
- **Real-time Prediction**: Allows users to input new digit images and see the model's predictions, showcasing the model's application beyond the dataset.

### 6. **Optimization and Further Steps**
- **Hyperparameter Tuning**: Adjusting parameters like learning rate, batch size, number of neurons, layers, and activation functions to improve performance.
- **Regularization**: Techniques like dropout, L1/L2 regularization can prevent overfitting.
- **Advanced Architectures**: Experimenting with convolutional neural networks (CNNs) can often yield better accuracy for image-related tasks.

## ALGORITHM:
1. Load the MNIST dataset and normalize pixel values between 0 and 1.
2. Flatten the images to a 1D array and encode labels as one-hot vectors.
3. Create an MLP with an input layer (784 neurons), two hidden layers (128 and 64 neurons, ReLU activation), and an output layer (10 neurons, softmax activation).
4. Compile the model using the 'adam' optimizer and 'categorical_crossentropy' loss.
5. Train the model for 5 epochs with a batch size of 32.
6. Plot training and validation accuracy over epochs.
7. Evaluate the model on the test dataset to obtain test accuracy.
8. Function to predict and display specific test digit images with their predicted and actual labels.
9. Provide an interactive function for predicting new digits and visualizing the model's prediction.

## PROGRAM:
```PYTHON
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten the images into a 1D array
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Convert target labels to categorical one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Create a Multilayer Perceptron model
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # 10 output neurons for 10 digit classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Visualize training history
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Function to predict and visualize results
def predict_and_visualize(index):
    prediction = model.predict(x_test[index].reshape(1, -1))
    predicted_label = np.argmax(prediction)
    actual_label = np.argmax(y_test[index])

    plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
    plt.title(f'Predicted: {predicted_label}, Actual: {actual_label}')
    plt.show()

# Visualize some predictions
predict_and_visualize(0)
predict_and_visualize(100)
predict_and_visualize(1000)

# Interactive prediction
def predict_new_digit(new_digit):
    new_digit = np.array(new_digit).reshape(1, -1) / 255.0
    prediction = model.predict(new_digit)
    predicted_label = np.argmax(prediction)
    
    plt.imshow(new_digit.reshape(28, 28), cmap='gray')
    plt.title(f'Predicted digit: {predicted_label}')
    plt.show()

# Example of interactive prediction
new_digit_input = x_test[0]  # Change this to test other digits
predict_new_digit(new_digit_input)
```
## OUTPUT:
### Model Training:
![image](https://github.com/Rithigasri/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/93427256/b865a908-9ad0-4be1-a495-a9f5e07d20f1)
### Training Vs Validation:
![image](https://github.com/Rithigasri/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/93427256/a88cf829-30a0-448d-8ce9-f355c1341872)
### Test Accuracy:
![image](https://github.com/Rithigasri/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/93427256/9dde2c6d-e5e2-4d64-a254-5eb60c7189de)

### Predictions:
![image](https://github.com/Rithigasri/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/93427256/6aa7dfe9-acee-4b33-924a-e828d5d54464)
![image](https://github.com/Rithigasri/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/93427256/71021d89-1852-4c13-b3c7-cb7064e97ebc)
![image](https://github.com/Rithigasri/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/93427256/5a051d8d-2026-45b5-abd3-7e6b30e7cec5)

## RESULT:
Thus, the implementation of Handwritten Digit Recognition using MLP Is executed successfully.
