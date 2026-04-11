# 🧠 TinyML Noise Robustness Benchmark — ESP32

Firmware **ESP32** untuk validasi ketahanan model **TinyML medis** (deteksi anomali tanda vital) terhadap berbagai jenis noise sensor. Sistem berjalan sepenuhnya di atas **TensorFlow Lite for Microcontrollers** dengan model Dense INT8 berukuran ±3 KB, tanpa sensor fisik, WiFi, maupun MQTT — hanya menggunakan dataset bawaan dan output Serial Monitor.

---

## 📋 Daftar Isi

- [Fitur Utama](#fitur-utama)
- [Demo & Preview](#demo--preview)
- [Arsitektur Sistem](#arsitektur-sistem)
- [Arsitektur Model Neural Network](#arsitektur-model-neural-network)
- [Alur Kerja Benchmark (Workflow)](#alur-kerja-benchmark-workflow)
- [7 Fase Noise Robustness](#7-fase-noise-robustness)
- [Struktur Direktori](#struktur-direktori)
- [Cara Kerja Inferensi](#cara-kerja-inferensi)
- [Incremental Learning (Header)](#incremental-learning-header)
- [Cara Build & Flash](#cara-build--flash)
- [Format Output Serial](#format-output-serial)
- [Dependensi](#dependensi)
- [Lisensi](#lisensi)

---

## ✨ Fitur Utama

- 🤖 **Deteksi anomali tanda vital** (SpO2, HR, Suhu) menggunakan model Dense 3→12→6→1 INT8
- 🔬 **7 skenario noise** — Baseline, Gaussian (3 level), Spike, Missing Data, Sensor Drift
- ⚡ **Inferensi ultra-cepat** < 1 ms per sampel pada ESP32 (APP_CPU / Core 1)
- 📊 **Akurasi tinggi** ≥ 94% pada kondisi noise ringan–sedang (referensi PC: 94.90–98.20%)
- 🎲 **Gaussian noise nyata** dari hardware RNG ESP32 (Box-Muller method)
- 💾 **Dataset tertanam** — 1000 sampel medis Z-score tersimpan di flash sebagai header C
- 📈 **Laporan otomatis** — ringkasan per fase & tabel final langsung di Serial Monitor
- 🔁 **Incremental Learning API** tersedia untuk pengembangan federated learning selanjutnya

---

## 🎬 Demo & Preview

> **Catatan:** GIF di bawah merekam output nyata firmware yang berjalan di atas ESP32 melalui Serial Monitor.  
> Jika gambar belum muncul, lihat panduan [`assets/README.md`](assets/README.md) untuk cara menambahkannya.

### 🖥️ Serial Monitor — Jalannya 7 Fase Benchmark

![Demo Serial Monitor](assets/demo-serial.gif)

<details>
<summary>Tidak melihat GIF? Klik untuk melihat contoh teks output</summary>

```
--- Phase 1/7: Baseline (No Noise)          ---
Akurasi Ref PC : 98.20%
  [ 200/1000] Acc: 98.5%  Last lat: 0.312 ms
  [ 400/1000] Acc: 98.2%  Last lat: 0.308 ms
  ...
╔══════════════════════════════════════════════════════════════╗
║  PHASE 0 SELESAI: Baseline (No Noise)                       ║
╠══════════════════════════════════════════════════════════════╣
║  Latency   : Min= 0.305 ms  Max= 0.420 ms  Avg= 0.312 ms  ║
║  ✅ Target < 100 ms : LULUS                                ║
╚══════════════════════════════════════════════════════════════╝
```

</details>

### 📊 Tabel Ringkasan Final — Hasil Benchmark 7 Fase

![Benchmark Results Dashboard](assets/benchmark-results.gif)

<details>
<summary>Tidak melihat GIF? Klik untuk melihat contoh teks output</summary>

```
=================================================================
       HARDWARE ROBUSTNESS TEST -- RINGKASAN FINAL
       Dataset: 1000 sampel medis (SpO2, HR, Temp Z-score)
       Model  : 3-12-6-1 Dense INT8 (3136 bytes = 3.06 KB)
=================================================================
  Skenario                     | Avg ms | <100ms | Acc ESP32 | Acc Ref
  ---------------------------  +--------+--------+-----------+---------
  Baseline (No Noise)         |  0.312 |   PASS |    98.20% |  98.20%
  Gaussian sigma=0.01         |  0.311 |   PASS |    98.10% |  98.10%
  Gaussian sigma=0.05         |  0.315 |   PASS |    96.20% |  96.20%
  Gaussian sigma=0.10         |  0.318 |   PASS |    94.90% |  94.90%
  Spike 5%                    |  0.313 |   PASS |    95.50% |  95.50%
  Missing 10%                 |  0.312 |   PASS |    96.70% |  96.70%
  Sensor Drift                |  0.310 |   PASS |    63.40% |  63.40%
=================================================================
```

</details>

---

## 🏗️ Arsitektur Sistem

Sistem berjalan di atas **FreeRTOS** dengan satu task utama benchmark pada Core 1:

```
┌─────────────────────────────────────────────────────────────┐
│                        ESP32 Chip                           │
│                                                             │
│   Core 0 (PRO_CPU)          Core 1 (APP_CPU)               │
│   ┌─────────────────┐       ┌──────────────────────────┐   │
│   │  FreeRTOS       │       │  taskNoiseBenchmark       │   │
│   │  Scheduler +    │       │  Prioritas : 5            │   │
│   │  System Tasks   │       │  Stack     : 32 KB        │   │
│   └─────────────────┘       │  Waktu     : ~7 fase      │   │
│                             └──────────┬─────────────────┘  │
│                                        │                    │
│   ┌────────────────────────────────────▼──────────────────┐ │
│   │  Flash (Read-Only)                                     │ │
│   │  ┌──────────────────┐   ┌────────────────────────┐    │ │
│   │  │  model.h         │   │  test_dataset.h         │    │ │
│   │  │  (3136 bytes     │   │  (1000 sampel ×         │    │ │
│   │  │   INT8 weights)  │   │   3 fitur Z-score)      │    │ │
│   │  └──────────────────┘   └────────────────────────┘    │ │
│   └────────────────────────────────────────────────────────┘ │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │  NVS Flash  (nvs_flash_init — required by TFLite)   │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
│   Output: UART0 → Serial Monitor (115200 baud)             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧬 Arsitektur Model Neural Network

Model **3-12-6-1 Fully Connected Dense** dengan kuantisasi INT8 penuh (*Post-Training Quantization*):

```
Input Layer        Hidden Layer 1    Hidden Layer 2    Output Layer
(3 neuron)        (12 neuron)        (6 neuron)        (1 neuron)

┌──────────┐       ┌──────────────┐  ┌──────────────┐  ┌─────────┐
│  SpO2_Z  │──────▶│              │  │              │  │         │
├──────────┤       │  Dense(12)   │─▶│  Dense(6)    │─▶│Dense(1) │
│   HR_Z   │──────▶│  Aktivasi:   │  │  Aktivasi:   │  │Aktivasi:│
├──────────┤       │  ReLU        │  │  ReLU        │  │  Tanh   │
│  Temp_Z  │──────▶│              │  │              │  │         │
└──────────┘       └──────────────┘  └──────────────┘  └────┬────┘
                                                             │
                                              ┌──────────────▼──────────────┐
                                              │  Output (tanh ∈ [-1, +1])  │
                                              │  > 0.0  → Anomaly  (1)     │
                                              │  ≤ 0.0  → Normal   (0)     │
                                              └─────────────────────────────┘

Parameter model : ~3136 bytes (INT8)
TFLite Arena    : 8 KB (SRAM)
Ops yang dipakai: FullyConnected, ReLU, Tanh, Dequantize
```

**Alur kuantisasi input → output:**
```
float feat[3]  →  kuantisasi int8  →  Invoke()  →  dequantize int8  →  float output
   (Z-score)      (scale + zp)       TFLite Micro    (scale + zp)       prediksi label
```

---

## 🔄 Alur Kerja Benchmark (Workflow)

```
                            ┌─────────────────────┐
                            │      app_main()      │
                            │  nvs_flash_init()    │
                            │  xTaskCreatePinnedTo │
                            │  Core(taskNoiseBench)│
                            └──────────┬──────────┘
                                       │
                            ┌──────────▼──────────┐
                            │  taskNoiseBenchmark  │
                            │  vTaskDelay(2000ms)  │
                            │  model_inference_    │
                            │  init()              │
                            └──────────┬──────────┘
                                       │
                    ┌──────────────────▼──────────────────┐
                    │         Loop 7 Fase (phase 0–6)     │
                    └──────────────────┬──────────────────┘
                                       │
                  ┌────────────────────▼────────────────────────┐
                  │         Untuk setiap fase:                   │
                  │                                              │
                  │  reset cumulative_drift = 0                  │
                  │                                              │
                  │  Loop 1000 sampel (s = 0..999)               │
                  │  ┌──────────────────────────────────────┐   │
                  │  │  feat[3] = test_features[s]          │   │
                  │  │  apply_noise(feat, phase)            │   │
                  │  │      ├─ Phase 0: no change           │   │
                  │  │      ├─ Phase 1: Gauss σ=0.01        │   │
                  │  │      ├─ Phase 2: Gauss σ=0.05        │   │
                  │  │      ├─ Phase 3: Gauss σ=0.10        │   │
                  │  │      ├─ Phase 4: Spike 5%            │   │
                  │  │      ├─ Phase 5: Missing 10%         │   │
                  │  │      └─ Phase 6: Drift +0.002/sample │   │
                  │  │  model_inference_run(feat)           │   │
                  │  │      → latency_us                    │   │
                  │  │      → raw_output (tanh)             │   │
                  │  │      → predicted_label (0/1)         │   │
                  │  │  akumulasi: lat_sum, lat_min/max      │   │
                  │  │  hitung: correct++ jika label cocok  │   │
                  │  │  yield setiap 100 sampel (watchdog)  │   │
                  │  └──────────────────────────────────────┘   │
                  │                                              │
                  │  print_phase_summary()  — cetak per fase    │
                  └────────────────────────────────────────────┘
                                       │
                            ┌──────────▼──────────┐
                            │  print_final_table() │
                            │  (ringkasan 7 fase)  │
                            └──────────┬──────────┘
                                       │
                            ┌──────────▼──────────┐
                            │   vTaskDelete(NULL)  │
                            └─────────────────────┘
```

---

## 📊 7 Fase Noise Robustness

| Fase | Nama Skenario | Deskripsi Noise | Ref Akurasi PC |
|------|--------------|-----------------|---------------|
| 0 | **Baseline (No Noise)** | Tidak ada noise | 98.20% |
| 1 | **Gaussian σ=0.01 (Ringan)** | Noise Gaussian kecil pada semua fitur | 98.10% |
| 2 | **Gaussian σ=0.05 (Sedang)** | Noise Gaussian sedang | 96.20% |
| 3 | **Gaussian σ=0.10 (Berat)** | Noise Gaussian besar | 94.90% |
| 4 | **Spike 5%** | 5% sampel mendapat z-score ekstrem (SpO2↓, HR↑, Temp↑) | 95.50% |
| 5 | **Missing Data 10%** | 10% sampel diimputasi ke 0 (z-score mean) | 96.70% |
| 6 | **Sensor Drift** | Drift kumulatif +0.002 per sampel | 63.40% |

**Kriteria kelulusan per fase:**
- ✅ Latensi rata-rata < 100 ms per sampel
- ✅ Akurasi ESP32 dalam toleransi ±5% dari referensi PC

---

## 📁 Struktur Direktori

```
tinyML-Federate-learning/
├── main/
│   ├── src/
│   │   ├── main.cpp                        # Entry point: nvs_flash_init + launch task
│   │   ├── test_macro.cpp                  # Utility test macro (reference only)
│   │   └── tinyml/
│   │       ├── model_inference.cpp         # TFLite Micro interpreter, kuantisasi INT8
│   │       └── noise_bench_task.cpp        # 7-fase benchmark, noise injection, report
│   ├── include/
│   │   ├── app_config.h                    # Konstanta: TEST_N_SAMPLES = 1000
│   │   ├── model_inference.h               # API: model_inference_init / _run
│   │   ├── noise_bench_task.h              # Deklarasi taskNoiseBenchmark
│   │   ├── tinyml/
│   │   │   └── incremental_learning.h      # API incremental/federated learning (future)
│   │   └── model/
│   │       └── test_dataset.h              # Dataset 1000 sampel + ref accuracy constants
│   ├── lib/
│   │   └── tinyml_model/
│   │       └── model.h                     # Binary INT8 model (3136 bytes, auto-gen)
│   ├── synthetic_autoencoder.tflite        # Model TFLite sumber (referensi)
│   ├── synthetic_autoencoder.h5            # Model Keras original (referensi)
│   ├── CMakeLists.txt                      # Registrasi komponen ESP-IDF
│   └── idf_component.yml                  # Dependensi komponen (TFLite Micro, dll.)
├── CMakeLists.txt                          # CMake root project
├── partitions.csv                          # Tabel partisi flash
├── sdkconfig.defaults                      # Konfigurasi default ESP-IDF
└── dependencies.lock                       # Lock file dependensi
```

---

## ⚙️ Cara Kerja Inferensi

### `model_inference_init()`

1. Memanggil `tflite::InitializeTarget()`
2. Memuat model dari `model[]` (array C di `lib/tinyml_model/model.h`)
3. Memvalidasi schema version TFLite
4. Mendaftarkan operator: `FullyConnected`, `ReLU`, `Tanh`, `Dequantize`
5. Mengalokasikan tensor arena 8 KB di SRAM
6. Menyimpan pointer input/output tensor

### `model_inference_run(const float feat[3])`

```
Input float[3]  →  kuantisasi ke int8  →  Invoke()  →  dequantize  →  InferenceResult
{ SpO2_Z,              (per-tensor              TFLite         (per-tensor     { latency_us,
  HR_Z,                 scale &                 Micro           scale &          raw_output,
  Temp_Z }              zero_point)                              zero_point)      label 0/1 }
```

| Field Output | Keterangan |
|-------------|-----------|
| `latency_us` | Waktu inferensi dalam mikrodetik (diukur via `esp_timer_get_time()`) |
| `raw_output` | Nilai tanh ∈ [-1.0, +1.0] setelah dequantize |
| `predicted_label` | `1` (Anomali) jika raw_output > 0.0, `0` (Normal) jika ≤ 0.0 |

---

## 🔁 Incremental Learning (Header)

File `include/tinyml/incremental_learning.h` menyediakan antarmuka untuk pengembangan **federated / incremental learning** ke depannya:

| Komponen | Keterangan |
|----------|-----------|
| `ExperienceReplay` | Ring buffer 20 sampel (input latent 16-dim + target 2-dim) |
| `TrainableLayer` | Layer FC kecil (16→2) dengan learning rate & L2 regularization |
| `add_to_replay_buffer()` | Menambah sampel baru ke buffer pengalaman |
| `train_on_buffer()` | Melatih layer FC dari buffer pengalaman |
| `save/load_weights_to/from_nvs()` | Persistensi bobot model ke NVS flash |

> **Catatan:** API ini tersedia sebagai header dan belum diaktifkan di firmware benchmark saat ini. Dirancang untuk integrasi federated learning on-device di iterasi berikutnya.

---

## 🚀 Cara Build & Flash

### Prasyarat

- [ESP-IDF](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/) versi ≥ 4.1
- CMake ≥ 3.16
- Python ≥ 3.8
- Board ESP32 + kabel USB

### Langkah Build

```bash
# 1. Clone repository
git clone https://github.com/herian-22/tinyML-Federate-learning.git
cd tinyML-Federate-learning

# 2. Set target ESP32
idf.py set-target esp32

# 3. Build project
idf.py build

# 4. Flash ke perangkat
idf.py -p /dev/ttyUSB0 flash

# 5. Monitor output serial (115200 baud)
idf.py -p /dev/ttyUSB0 monitor
```

> Tidak ada konfigurasi tambahan yang diperlukan. Benchmark akan berjalan otomatis setelah boot.

---

## 📊 Format Output Serial

### Progress per 200 sampel (dalam setiap fase)

```
--- Phase 1/7: Baseline (No Noise)          ---
Akurasi Ref PC : 98.20%
  [ 200/1000] Acc: 98.5%  Last lat: 0.312 ms
  [ 400/1000] Acc: 98.2%  Last lat: 0.308 ms
  ...
```

### Ringkasan per fase

```
╔══════════════════════════════════════════════════════════════╗
║  PHASE 0 SELESAI: Baseline (No Noise)                       ║
╠══════════════════════════════════════════════════════════════╣
║  Sampel    : 1000                                           ║
║  Latency   : Min= 0.305 ms  Max= 0.420 ms  Avg= 0.312 ms  ║
║  ✅ Target < 100 ms : LULUS                                ║
╠══════════════════════════════════════════════════════════════╣
║  Akurasi ESP32  :  98.20% (982/1000 benar)                 ║
║  Akurasi Ref PC :  98.20% (noise_robustness_report.md)     ║
║  ✅ Akurasi dalam toleransi ±5% referensi                  ║
╚══════════════════════════════════════════════════════════════╝
```

### Tabel ringkasan final (7 fase)

```
=================================================================
       HARDWARE ROBUSTNESS TEST -- RINGKASAN FINAL
       Dataset: 1000 sampel medis (SpO2, HR, Temp Z-score)
       Model  : 3-12-6-1 Dense INT8 (3136 bytes = 3.06 KB)
=================================================================
  Skenario                     | Avg ms | <100ms | Acc ESP32 | Acc Ref
  ---------------------------  +--------+--------+-----------+---------
  Baseline (No Noise)         |  0.312 |   PASS |    98.20% |  98.20%
  Gaussian sigma=0.01         |  0.311 |   PASS |    98.10% |  98.10%
  Gaussian sigma=0.05         |  0.315 |   PASS |    96.20% |  96.20%
  Gaussian sigma=0.10         |  0.318 |   PASS |    94.90% |  94.90%
  Spike 5%                    |  0.313 |   PASS |    95.50% |  95.50%
  Missing 10%                 |  0.312 |   PASS |    96.70% |  96.70%
  Sensor Drift                |  0.310 |   PASS |    63.40% |  63.40%
=================================================================
```

---

## 📦 Dependensi

### Komponen Eksternal (idf_component.yml)

| Komponen | Versi | Fungsi |
|----------|-------|--------|
| `espressif/esp-tflite-micro` | `^1.0.0` | TensorFlow Lite for Microcontrollers |

### Komponen Built-in ESP-IDF

`nvs_flash` · `esp_timer` · `esp_system` · `freertos` · `esp_random`

---

## 📄 Lisensi

Proyek ini dikembangkan untuk keperluan penelitian dan edukasi.

---

<div align="center">
  <p>Dikembangkan dengan ❤️ untuk penelitian TinyML & Federated Learning pada perangkat tepi</p>
</div>
