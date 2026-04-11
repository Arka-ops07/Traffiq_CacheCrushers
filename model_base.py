import tensorflow as tf
from tensorflow.keras import layers, models

def build_traffiq_model(input_shape=(224, 224, 3)):
    """
    Builds a CNN architecture for autonomous steering and throttle.
    """
    model = models.Sequential([
        # Normalization layer
        layers.Rescaling(1./255, input_shape=input_shape),
        
        # Convolutional feature extractors
        layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu'),
        layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
        layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten and Dense layers for decision making
        layers.Flatten(),
        layers.Dense(100, activation='relu'),
        layers.Dropout(0.1), # Helps prevent overfitting
        layers.Dense(50, activation='relu'),
        layers.Dense(10, activation='relu'),
        
        # Output layer: 2 neurons (Speed/Throttle and Steering Direction)
        # Using 'tanh' so outputs are between -1 and 1
        layers.Dense(2, activation='tanh') 
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def convert_to_tflite(h5_model_path, tflite_save_path):
    """
    Converts a trained Keras .h5 model to a lightweight .tflite model.
    """
    model = tf.keras.models.load_model(h5_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optional but highly recommended: Quantization shrinks the model size and speeds up Pi inference
    converter.optimizations = [tf.lite.Optimize.DEFAULT] 
    
    tflite_model = converter.convert()
    
    with open(tflite_save_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Successfully converted and saved to {tflite_save_path}")

if __name__ == "__main__":
    # 1. Build the model
    model = build_traffiq_model()
    model.summary()
    
    # --- YOUR TRAINING LOOP GOES HERE ---
    # model.fit(X_train, y_train, epochs=10)
    # model.save("traffiq_model.h5")
    
    # 2. Convert to TFLite (Uncomment after training)
    # convert_to_tflite("traffiq_model.h5", "traffiq_model.tflite")