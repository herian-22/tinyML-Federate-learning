/**
 * @file main.cpp
 * @brief ESP32 TinyML Noise Robustness Benchmark Entry Point
 *
 * Minimal firmware untuk validasi hardware:
 *   - Tidak ada sensor fisik
 *   - Tidak ada WiFi / MQTT / Web Server
 *   - Menjalankan 6-fase noise robustness benchmark langsung dari dataset
 *   - Output hanya via Serial Monitor (115200 baud)
 */

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"
#include "esp_log.h"
#include "app_config.h"
#include "noise_bench_task.h"

static const char *TAG = "Main";

extern "C" void app_main(void) {
    // NVS diperlukan oleh TFLite Micro stack
    esp_err_t nvs_ret = nvs_flash_init();
    if (nvs_ret == ESP_ERR_NVS_NO_FREE_PAGES ||
        nvs_ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        nvs_ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(nvs_ret);

    ESP_LOGI(TAG, "=================================================");
    ESP_LOGI(TAG, "  TinyML Noise Robustness Benchmark");
    ESP_LOGI(TAG, "  Model   : 3-12-6-1 Dense INT8 (3136 bytes)");
    ESP_LOGI(TAG, "  Dataset : %d sampel (SpO2, HR, Temp Z-score)", TEST_N_SAMPLES);
    ESP_LOGI(TAG, "  Fases   : 6 (Baseline + 3 Gaussian + Spike + Missing)");
    ESP_LOGI(TAG, "=================================================");

    // Launch benchmark task on Core 1, high stack (TFLite needs ~8KB arena + overhead)
    xTaskCreatePinnedToCore(
        taskNoiseBenchmark,
        "NoiseBench",
        32768,   // 32 KB stack
        NULL,
        5,
        NULL,
        1        // Core 1 (APP_CPU)
    );
}