/**
 * @file model_inference.cpp
 * @brief Direct TFLite Micro inference for real 3-12-6-1 INT8 model.
 *
 * Model: main/lib/tinyml_model/model.h
 * Architecture: Dense 3 → 12 (ReLU) → 6 (ReLU) → 1 (Tanh)
 * Quantization: Full INT8 (Post-Training Quantization)
 * Input : float[3]  = { SpO2_ZScore, HR_ZScore, Temp_ZScore }
 * Output: float     = raw tanh output in [-1.0, +1.0]
 *                     > 0.0 → Anomaly (1), <= 0.0 → Normal (0)
 */

#include "model_inference.h"
#include "model.h"  // from lib/tinyml_model/model.h (model[] + model_len)

#include "esp_log.h"
#include "esp_timer.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char *TAG = "ModelInference";

// ---------------------------------------------------------------------------
// TFLite Micro globals
// ---------------------------------------------------------------------------
namespace {
    // Arena: 3-12-6-1 is tiny — 8 KB is more than enough
    constexpr int kTensorArenaSize = 8 * 1024;
    static uint8_t tensor_arena[kTensorArenaSize];

    static tflite::MicroMutableOpResolver<4> resolver;
    static const tflite::Model               *tfl_model  = nullptr;
    static tflite::MicroInterpreter          *interpreter = nullptr;
    static TfLiteTensor                      *input_tensor  = nullptr;
    static TfLiteTensor                      *output_tensor = nullptr;
    static bool                               initialized    = false;
}

// ---------------------------------------------------------------------------
// Public: initialize once
// ---------------------------------------------------------------------------
bool model_inference_init(void) {
    if (initialized) return true;

    tflite::InitializeTarget();

    tfl_model = tflite::GetModel(model);
    if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema mismatch: got %u, expected %d",
                 tfl_model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    // Register only the ops needed for 3-12-6-1 Dense (FullyConnected + Tanh)
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddTanh();
    resolver.AddDequantize();

    static tflite::MicroInterpreter static_interpreter(
        tfl_model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed");
        return false;
    }

    input_tensor  = interpreter->input(0);
    output_tensor = interpreter->output(0);

    ESP_LOGI(TAG, "Model ready. Input shape: [%d, %d]  Output shape: [%d, %d]",
             input_tensor->dims->data[0], input_tensor->dims->data[1],
             output_tensor->dims->data[0], output_tensor->dims->data[1]);
    ESP_LOGI(TAG, "Model size: %u bytes (%.2f KB)",
             model_len, model_len / 1024.0f);

    initialized = true;
    return true;
}

// ---------------------------------------------------------------------------
// Public: run one inference
// Returns: InferenceResult with latency_us, raw_output, predicted_label
// ---------------------------------------------------------------------------
InferenceResult model_inference_run(const float feat[3]) {
    InferenceResult result = {0, 0.0f, 0};

    if (!initialized) {
        ESP_LOGE(TAG, "Not initialized!");
        result.predicted_label = -1;
        return result;
    }

    // Quantize float input → int8 using tensor scale/zero_point
    float   in_scale      = input_tensor->params.scale;
    int32_t in_zero_point = input_tensor->params.zero_point;

    for (int i = 0; i < 3; i++) {
        int32_t q = (int32_t)roundf(feat[i] / in_scale) + in_zero_point;
        // Clamp to int8 range
        if (q < -128) q = -128;
        if (q >  127) q =  127;
        input_tensor->data.int8[i] = (int8_t)q;
    }

    // Run inference and time it
    int64_t t0 = esp_timer_get_time();
    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke() failed");
        result.predicted_label = -1;
        return result;
    }
    int64_t t1 = esp_timer_get_time();
    result.latency_us = (uint32_t)(t1 - t0);

    // Dequantize int8 output → float
    float   out_scale      = output_tensor->params.scale;
    int32_t out_zero_point = output_tensor->params.zero_point;
    int8_t  raw_out        = output_tensor->data.int8[0];
    result.raw_output      = (raw_out - out_zero_point) * out_scale;

    // Classify: tanh output > 0 → Anomaly=1, ≤ 0 → Normal=0
    result.predicted_label = (result.raw_output > 0.0f) ? 1 : 0;

    return result;
}
