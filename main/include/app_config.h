#pragma once

/**
 * @file app_config.h
 * @brief Minimal configuration for TinyML Noise Robustness Benchmark.
 *
 * Firmware ini HANYA menjalankan benchmark dataset → model → Serial output.
 * Tidak ada sensor fisik, WiFi, atau MQTT.
 */

// Test dataset size (harus cocok dengan yang di-generate generate_test_dataset_header.py)
#define TEST_N_SAMPLES 1000
