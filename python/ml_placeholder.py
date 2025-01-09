import tensorflow as tf

def dummy_ml_model():
    # A trivial TensorFlow “stub”
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

if __name__ == "__main__":
    model = dummy_ml_model()
    print("Dummy ML model created:", model.summary())
