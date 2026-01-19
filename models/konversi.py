import tensorflow as tf
import numpy as np

# === 1. Load model asli .tflite (float) ===
tflite_model_path = "model_ambulance_siren.tflite"

# === 2. Buat model Keras dummy sesuai input shape kamu ===
# (misal input: 1x24, ganti kalau beda)
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(24,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# === 3. Buat converter dari model dummy ===
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_data_gen():
    for _ in range(100):
        data = np.random.rand(1, 24).astype(np.float32)
        yield [data]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# === 4. Konversi ke model INT8 ===
quantized_model = converter.convert()

# === 5. Simpan hasil quantized ===
with open("model_ambulance_siren_int8.tflite", "wb") as f:
    f.write(quantized_model)

print("✅ model_ambulance_siren_int8.tflite berhasil dibuat (INT8 quantized)")
