import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

# Load the TFLite model from the current directory
interpreter = tf.lite.Interpreter(model_path='cifar10_model.tflite')
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess a sample input image from the CIFAR-10 dataset
(_, _), (x_test, y_test) = cifar10.load_data()
x_test = x_test / 255.0  # Normalize the images to [0, 1]
input_data = np.expand_dims(x_test[0], axis=0).astype(np.float32)  # Expand dimensions to match the input tensor shape

# Set the input tensor to the sample image
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run the model
interpreter.invoke()

# Get the output tensor (the model's prediction)
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(output_data)

# Print the predicted class and the actual class
print('Predicted class:', predicted_class)
print('Actual class:', y_test[0][0])
