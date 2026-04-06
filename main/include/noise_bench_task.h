#pragma once

/**
 * @brief 6-Phase Noise Robustness Benchmark Task.
 *
 * Mereplikasi skenario uji dari main/lib/noise_robustness_report.md
 * menggunakan dataset medis langsung (SpO2, HR, Temp Z-score).
 *
 * Tidak memerlukan sensor fisik.
 */
void taskNoiseBenchmark(void *pvParameters);
