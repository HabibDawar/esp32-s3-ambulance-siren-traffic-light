#include <Arduino.h>
#include <I2S.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model_ambulance_siren.h"

// ===== CONFIGURATION =====
#define SAMPLE_RATE 16000
#define RECORD_MS 1000
#define NUM_SAMPLES (SAMPLE_RATE * RECORD_MS / 1000)
#define NUM_FEATURES 40

// ESP32-S3 Traffic Light LED Pins
#define RED_LED 4
#define YELLOW_LED 5
#define GREEN_LED 6

// ESP32-S3 I2S Pin Configuration
#define I2S_WS 15
#define I2S_SCK 13
#define I2S_SD 10

// ===== TENSORFLOW LITE SETUP =====
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  constexpr int kTensorArenaSize = 50 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}

// ===== AUDIO BUFFERS =====
int16_t audio_buffer[NUM_SAMPLES];
float feature_buffer[NUM_FEATURES];

// ===== FUNCTION DECLARATIONS =====
bool recordAudio();
void extractFeatures();
void controlTrafficLight(bool ambulance_detected);

// ===== SETUP =====
void setup() {
  // Initialize LED pins
  pinMode(RED_LED, OUTPUT);
  pinMode(YELLOW_LED, OUTPUT);
  pinMode(GREEN_LED, OUTPUT);
  
  // Start with green light
  digitalWrite(RED_LED, LOW);
  digitalWrite(YELLOW_LED, LOW);
  digitalWrite(GREEN_LED, HIGH);

  // Initialize serial communication
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }
  Serial.println("ESP32-S3 Traffic Light Controller Starting...");

  // Initialize I2S
  I2S.setAllPins(-1, I2S_SCK, I2S_WS, I2S_SD, -1);
  if (!I2S.begin(I2S_PHILIPS_MODE, SAMPLE_RATE, 16)) {
    Serial.println("Failed to initialize I2S!");
    while (1) {
      // Blink red LED to indicate error
      digitalWrite(RED_LED, !digitalRead(RED_LED));
      delay(500);
    }
  }
  Serial.println("I2S Initialized successfully");

  // Initialize TensorFlow Lite
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure
  model = tflite::GetModel(model_ambulance_siren);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model version mismatch. Expected: ");
    Serial.print(TFLITE_SCHEMA_VERSION);
    Serial.print(", Got: ");
    Serial.println(model->version());
    return;
  }

  // Instantiate operations resolver
  static tflite::AllOpsResolver resolver;

  // Build interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Print tensor information
  Serial.println("=== Tensor Information ===");
  Serial.print("Input dimensions: ");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    Serial.print(" ");
  }
  Serial.println();
  
  Serial.print("Output dimensions: ");
  for (int i = 0; i < output->dims->size; i++) {
    Serial.print(output->dims->data[i]);
    Serial.print(" ");
  }
  Serial.println();

  Serial.println("Setup complete - Ready for inference");
}

// ===== MAIN LOOP =====
void loop() {
  static unsigned long lastDetectionTime = 0;
  const unsigned long detectionCooldown = 10000; // 10 seconds

  // Record audio
  if (recordAudio()) {
    // Extract features
    extractFeatures();
    
    // Copy features to input tensor
    // Adjust based on your model's expected input shape
    for (int i = 0; i < NUM_FEATURES && i < input->dims->data[1]; i++) {
      input->data.f[i] = feature_buffer[i];
    }
    
    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      Serial.println("Invoke failed!");
      return;
    }
    
    // Process results
    // Adjust based on your model's output format
    float no_siren_confidence = output->data.f[0];
    float siren_confidence = output->data.f[1];
    
    bool ambulance_detected = (siren_confidence > 0.7) && (siren_confidence > no_siren_confidence);
    
    Serial.print("No Siren: ");
    Serial.print(no_siren_confidence, 3);
    Serial.print(" | Siren: ");
    Serial.print(siren_confidence, 3);
    Serial.print(" | Detected: ");
    Serial.println(ambulance_detected ? "YES" : "NO");
    
    // Control traffic light
    if (ambulance_detected && (millis() - lastDetectionTime > detectionCooldown)) {
      lastDetectionTime = millis();
      controlTrafficLight(true);
    }
  }
  
  delay(500); // Wait between inferences
}

// ===== FUNCTION DEFINITIONS =====

bool recordAudio() {
  int samples_read = 0;
  unsigned long startTime = millis();
  
  while (samples_read < NUM_SAMPLES) {
    int16_t sample[2]; // Stereo input
    int bytes_read = I2S.read((void*)sample, sizeof(sample));
    
    if (bytes_read == sizeof(sample)) {
      // Use only left channel (sample[0])
      audio_buffer[samples_read] = sample[0];
      samples_read++;
    } else if (bytes_read == 0) {
      // No data available
      if (millis() - startTime > 2000) {
        Serial.println("Timeout reading audio data");
        return false;
      }
      delay(1);
    } else {
      Serial.println("Error reading from I2S");
      return false;
    }
  }
  
  return true;
}

void extractFeatures() {
  // Simple feature extraction - replace with your actual feature extraction
  // This is just a placeholder
  
  // Calculate RMS energy
  long long sum_squares = 0;
  for (int i = 0; i < NUM_SAMPLES; i++) {
    sum_squares += (long long)audio_buffer[i] * audio_buffer[i];
  }
  float rms = sqrt(sum_squares / (float)NUM_SAMPLES);
  
  // Normalize and create simple features
  float normalized_rms = rms / 10000.0f; // Adjust scaling as needed
  if (normalized_rms > 1.0f) normalized_rms = 1.0f;
  
  // Create feature vector (simplified)
  for (int i = 0; i < NUM_FEATURES; i++) {
    feature_buffer[i] = normalized_rms * (1.0f + 0.1f * sin(i * 0.5f));
  }
  
  // Print debug info occasionally
  static unsigned long lastPrint = 0;
  if (millis() - lastPrint > 5000) {
    lastPrint = millis();
    Serial.print("Audio RMS: ");
    Serial.println(rms);
  }
}

void controlTrafficLight(bool ambulance_detected) {
  if (ambulance_detected) {
    Serial.println("AMBULANCE DETECTED! Activating emergency mode...");
    
    // Sequence: Green -> Yellow -> Red
    digitalWrite(GREEN_LED, LOW);
    digitalWrite(YELLOW_LED, HIGH);
    delay(2000);
    
    digitalWrite(YELLOW_LED, LOW);
    digitalWrite(RED_LED, HIGH);
    
    // Keep red light for emergency vehicle
    Serial.println("TRAFFIC STOPPED for ambulance");
    delay(10000); // 10 seconds red light
    
    // Sequence: Red -> Yellow -> Green
    digitalWrite(RED_LED, LOW);
    digitalWrite(YELLOW_LED, HIGH);
    delay(2000);
    
    digitalWrite(YELLOW_LED, LOW);
    digitalWrite(GREEN_LED, HIGH);
    
    Serial.println("Returning to normal operation");
  } else {
    // Normal operation - green light
    digitalWrite(RED_LED, LOW);
    digitalWrite(YELLOW_LED, LOW);
    digitalWrite(GREEN_LED, HIGH);
  }
}