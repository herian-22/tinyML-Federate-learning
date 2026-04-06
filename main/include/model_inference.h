#pragma once

#include <stdint.h>
#include <stdbool.h>

/**
 * Result from a single inference run.
 */
typedef struct {
    uint32_t latency_us;     /**< Inference time in microseconds */
    float    raw_output;     /**< Raw tanh output in [-1.0, +1.0] */
    int      predicted_label; /**< 0=Normal, 1=Anomaly, -1=Error */
} InferenceResult;

/**
 * Initialize the TFLite Micro interpreter once at startup.
 * @return true if successful.
 */
bool model_inference_init(void);

/**
 * Run one inference on a 3-feature z-score input.
 * @param feat float[3] = { SpO2_ZScore, HR_ZScore, Temp_ZScore }
 * @return InferenceResult
 */
InferenceResult model_inference_run(const float feat[3]);
