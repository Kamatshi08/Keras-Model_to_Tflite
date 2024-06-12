import tensorflow as tf

# Load the trained Keras model saved in the current directory
model = tf.keras.models.load_model('cifar10_model.keras')

# Create a TFLite converter from the Keras model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Convert the model to TFLite format
tflite_model = converter.convert()

# Save the converted TFLite model in the current directory
with open('cifar10_model.tflite', 'wb') as f:
    f.write(tflite_model)
