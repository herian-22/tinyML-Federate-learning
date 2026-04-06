/**
 * @file noise_bench_task.cpp
 * @brief 6-Phase Noise Robustness Benchmark for ESP32 Hardware Validation
 *
 * Mereplikasi LANGSUNG skenario uji dari dokumentasi:
 *   main/lib/noise_robustness_report.md
 *
 * Data input : test_dataset.h (1000 sampel dari data CSV)
 * Model      : tinyml_model/model.h  (3-12-6-1, INT8, 3136 bytes)
 * Input      : SpO2_ZScore, HR_ZScore, Temp_ZScore  (3 fitur, sudah Z-score)
 * Output     : tanh → > 0.0 = Anomaly, ≤ 0.0 = Normal
 *
 * 6 Fase (sesuai noise_robustness_report.md):
 *   Phase 0 — Baseline (no noise)          Ref Acc: 98.74%
 *   Phase 1 — Gaussian σ=0.01 (Ringan)     Ref Acc: 98.71%
 *   Phase 2 — Gaussian σ=0.05 (Sedang)     Ref Acc: 97.23%
 *   Phase 3 — Gaussian σ=0.10 (Berat)      Ref Acc: 94.98%
 *   Phase 4 — Spike Noise 5%               Ref Acc: 95.77%
 *   Phase 5 — Missing Data 10% (impute 0)  Ref Acc: 96.38%
 *
 * Catatan referensi latensi (PC float32, 1000 sampel, tidak relevan untuk
 * perbandingan ESP32 — hanya sebagai konteks dokumentasi).
 */

#include "noise_bench_task.h"
#include "model_inference.h"
#include "app_config.h"
#include "model/test_dataset.h"

#include "esp_log.h"
#include "esp_timer.h"
#include "esp_random.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

static const char *TAG = "NoiseBench";
static float cumulative_drift = 0.0f;

// ---------------------------------------------------------------------------
// Phase metadata (sesuai noise_robustness_report.md)
// ---------------------------------------------------------------------------
typedef struct {
    const char *name;
    float       ref_acc_pct;
    float       ref_lat_pc_ms;  // hanya referensi, bukan target untuk ESP32
} PhaseInfo;

static const PhaseInfo PHASES[7] = {
    { "Baseline (No Noise)         ", REF_ACC_BASELINE,    REF_LAT_BASELINE_MS    },
    { "Gaussian  sigma=0.01 Ringan ", REF_ACC_GAUSS_LIGHT, REF_LAT_GAUSS_LIGHT_MS },
    { "Gaussian  sigma=0.05 Sedang ", REF_ACC_GAUSS_MED_0_05,   REF_LAT_GAUSS_MED_MS_0_05   },
    { "Gaussian  sigma=0.10 Berat  ", REF_ACC_GAUSS_HEAVY, REF_LAT_GAUSS_HEAVY_MS },
    { "Spike     5%  glitch         ", REF_ACC_SPIKE,       REF_LAT_SPIKE_MS       },
    { "Missing   10% impute mean=0  ", REF_ACC_MISSING,     REF_LAT_MISSING_MS     },
    { "Sensor Drift (+0.002/sample) ", REF_ACC_DRIFT,       REF_LAT_DRIFT_MS       },
};


// ---------------------------------------------------------------------------
// Box-Muller Gaussian noise via ESP32 hardware RNG
// ---------------------------------------------------------------------------
static float gaussian_noise(float sigma) {
    float u1 = ((float)(esp_random() & 0x7FFFFFFF) + 1.0f) / (float)0x80000000;
    float u2 = ((float)(esp_random() & 0x7FFFFFFF) + 1.0f) / (float)0x80000000;
    float z  = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
    return sigma * z;
}

// ---------------------------------------------------------------------------
// Apply phase-specific noise to a copy of the 3-feature sample
// ---------------------------------------------------------------------------
static void apply_noise(float feat[3], int phase) {
    switch (phase) {
        case 0:  // Baseline — no change
            break;
        case 1:  // Gaussian sigma=0.01
            feat[0] += gaussian_noise(0.01f);
            feat[1] += gaussian_noise(0.01f);
            feat[2] += gaussian_noise(0.01f);
            break;
        case 2:  // Gaussian sigma=0.05
            feat[0] += gaussian_noise(0.05f);
            feat[1] += gaussian_noise(0.05f);
            feat[2] += gaussian_noise(0.05f);
            break;
        case 3:  // Gaussian sigma=0.10
            feat[0] += gaussian_noise(0.10f);
            feat[1] += gaussian_noise(0.10f);
            feat[2] += gaussian_noise(0.10f);
            break;
        case 4:  // Spike: 5% probability — extreme z-score values
            if ((esp_random() % 100) < 5) {
                feat[0] = -3.5f;  // SpO2 sangat rendah (z-score)
                feat[1] =  4.5f;  // HR tachycardia parah
                feat[2] =  3.5f;  // Suhu demam tinggi
            }
            break;
        case 5:  // Missing data: 10% probability — impute mean (z-score = 0)
            if ((esp_random() % 100) < 10) {
                feat[0] = 0.0f;
                feat[1] = 0.0f;
                feat[2] = 0.0f;
            }
            break;
        case 6: // Contoh Noise baru: Sensor Drift (Makin lama makin ngaco)
            cumulative_drift += 0.002f; // Bertambah sedikit demi sedikit
            feat[0] += cumulative_drift; 
            feat[1] += cumulative_drift;
            feat[2] += cumulative_drift;
            break;
        
        default:
            break;
    }
}

// ---------------------------------------------------------------------------
// Print per-phase summary banner
// ---------------------------------------------------------------------------
static void print_phase_summary(int phase,
                                uint32_t n,
                                float avg_ms, float min_ms, float max_ms,
                                int correct) {
    float acc = 100.0f * (float)correct / (float)n;
    const PhaseInfo *p = &PHASES[phase];
    bool lat_ok = (avg_ms < 100.0f);
    bool acc_ok = (acc >= (p->ref_acc_pct - 5.0f));

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  PHASE %d SELESAI: %-42s║\n", phase, p->name);
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  Sampel    : %-4" PRIu32 "                                           ║\n", n);
    printf("║  Latency   : Min=%6.3f ms  Max=%6.3f ms  Avg=%6.3f ms  ║\n",
           min_ms, max_ms, avg_ms);
    if (lat_ok) {
        printf("║  ✅ Target < 100 ms : LULUS                                ║\n");
    } else {
        printf("║  ❌ Target < 100 ms : GAGAL                                ║\n");
    }
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  Akurasi ESP32  : %6.2f%% (%u/%" PRIu32 " benar)              ║\n",
           acc, (unsigned)correct, n);
    printf("║  Akurasi Ref PC : %6.2f%% (noise_robustness_report.md)    ║\n",
           p->ref_acc_pct);
    if (acc_ok) {
        printf("║  ✅ Akurasi dalam toleransi ±5%% referensi                  ║\n");
    } else {
        printf("║  ⚠️  Akurasi di bawah toleransi referensi                   ║\n");
    }
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

// ---------------------------------------------------------------------------
// Print final summary table
// ---------------------------------------------------------------------------
static void print_final_table(float acc[7], float avg_lat[7],
                               float min_lat[7], float max_lat[7]) {
    printf("\n");
    printf("=================================================================\n");
    printf("       HARDWARE ROBUSTNESS TEST -- RINGKASAN FINAL\n");
    printf("       Dataset: %d sampel medis (SpO2, HR, Temp Z-score)\n", TEST_N_SAMPLES);
    printf("       Model  : 3-12-6-1 Dense INT8 (%u bytes = %.2f KB)\n",
           3136u, 3136.0f / 1024.0f);
    printf("=================================================================\n");
    printf("  %-28s | Avg ms | <100ms | Acc ESP32 | Acc Ref\n",
           "Skenario");
    printf("  %-28s-+--------+--------+-----------+---------\n",
           "----------------------------");

    const char *labels[7] = {
        "Baseline (No Noise)    ",
        "Gaussian sigma=0.01    ",
        "Gaussian sigma=0.05    ",
        "Gaussian sigma=0.10    ",
        "Spike 5%%               ",
        "Missing 10%%            ",
        "Sensor Drift           ",
    };
    for (int i = 0; i < 7; i++) {
        printf("  %-28s | %6.3f | %6s | %8.2f%% | %6.2f%%\n",
               labels[i],
               avg_lat[i],
               (avg_lat[i] < 100.0f) ? "PASS" : "FAIL",
               acc[i],
               PHASES[i].ref_acc_pct);
    }
    printf("=================================================================\n");
    printf("\n");
    printf("[DONE] Salin tabel di atas ke laporan tesis kamu.\n");
    printf("[DONE] Nonaktifkan NOISE_TEST_MODE di app_config.h setelah selesai.\n");
    printf("\n");
}

// ---------------------------------------------------------------------------
// FreeRTOS task entry point
// ---------------------------------------------------------------------------
void taskNoiseBenchmark(void *pvParameters) {
    ESP_LOGI(TAG, "Noise Benchmark Task dimulai.");

    // Tunggu sistem stabil
    vTaskDelay(pdMS_TO_TICKS(2000));

    // Inisialisasi model
    printf("\n");
    printf("=================================================================\n");
    printf("  UJI KETAHANAN NOISE -- Replikasi noise_robustness_report.md\n");
    printf("  Dataset : %d sampel (SpO2_Z, HR_Z, Temp_Z + Anomaly_Label)\n",
           TEST_N_SAMPLES);
    printf("  Model   : 3-12-6-1 Dense INT8 (3136 bytes, dari tinyml_model/)\n");
    printf("  Fase    : 7 (Baseline + 3 Gaussian + Spike + Missing Data + Drift)\n");
    printf("=================================================================\n\n");

    if (!model_inference_init()) {
        ESP_LOGE(TAG, "GAGAL inisialisasi model. Task dihentikan.");
        vTaskDelete(NULL);
        return;
    }

    float final_acc[7];
    float final_avg[7];
    float final_min[7];
    float final_max[7];

    // Jalankan 7 fase
    for (int phase = 0; phase < 7; phase++) {
        printf("--- Phase %d/7: %s ---\n", phase + 1, PHASES[phase].name);
        
        // Reset drift di awal setiap fase (terutama kritikal untuk Phase 6)
        cumulative_drift = 0.0f;

        printf("Akurasi Ref PC : %.2f%%\n", PHASES[phase].ref_acc_pct);

        uint32_t lat_min_us = UINT32_MAX;
        uint32_t lat_max_us = 0;
        uint64_t lat_sum_us = 0;
        int      correct    = 0;

        for (int s = 0; s < TEST_N_SAMPLES; s++) {
            // Salin fitur agar tidak memodifikasi dataset asli
            float feat[3] = {
                test_features[s][0],
                test_features[s][1],
                test_features[s][2],
            };

            // Terapkan noise sesuai fase
            apply_noise(feat, phase);

            // Jalankan inferensi
            InferenceResult res = model_inference_run(feat);

            // Akumulasi latensi
            lat_sum_us += res.latency_us;
            if (res.latency_us < lat_min_us) lat_min_us = res.latency_us;
            if (res.latency_us > lat_max_us) lat_max_us = res.latency_us;

            // Hitung akurasi
            if (res.predicted_label == (int)test_labels[s]) correct++;

            // Progress setiap 200 sampel
            if ((s + 1) % 200 == 0) {
                float acc_now = 100.0f * (float)correct / (float)(s + 1);
                float lat_now = (float)res.latency_us / 1000.0f;
                printf("  [%4d/%d] Acc: %.1f%%  Last lat: %.3f ms\n",
                        s + 1, TEST_N_SAMPLES, acc_now, lat_now);
            }

            // Yield setiap 100 sampel agar watchdog tidak reset
            if ((s % 100) == 0) {
                vTaskDelay(1);
            }
        }

        float avg_ms = (float)lat_sum_us / TEST_N_SAMPLES / 1000.0f;
        float min_ms = (float)lat_min_us / 1000.0f;
        float max_ms = (float)lat_max_us / 1000.0f;

        final_acc[phase] = 100.0f * (float)correct / TEST_N_SAMPLES;
        final_avg[phase] = avg_ms;
        final_min[phase] = min_ms;
        final_max[phase] = max_ms;

        print_phase_summary(phase, TEST_N_SAMPLES, avg_ms, min_ms, max_ms, correct);
        vTaskDelay(pdMS_TO_TICKS(500));
    }

    print_final_table(final_acc, final_avg, final_min, final_max);

    ESP_LOGI(TAG, "Benchmark selesai.");
    vTaskDelete(NULL);
}
