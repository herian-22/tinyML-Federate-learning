#ifndef INCREMENTAL_LEARNING_H
#define INCREMENTAL_LEARNING_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define REPLAY_BUFFER_SIZE 20
#define FC_INPUT_DIM 16   // Assuming bottleneck is 16
#define FC_OUTPUT_DIM 2    // Temperature and Humidity

typedef struct {
    float input[FC_INPUT_DIM];
    float target[FC_OUTPUT_DIM];
} ExperienceSample;

typedef struct {
    ExperienceSample samples[REPLAY_BUFFER_SIZE];
    int head;
    int count;
} ExperienceReplay;

typedef struct {
    float weights[FC_OUTPUT_DIM][FC_INPUT_DIM];
    float biases[FC_OUTPUT_DIM];
    float learning_rate;
    float l2_reg;
} TrainableLayer;

// --- API ---
void init_incremental_learning();
void add_to_replay_buffer(const float* latent, const float* target);
void train_on_buffer(TrainableLayer* layer, ExperienceReplay* buffer);
void fc_forward(const TrainableLayer* layer, const float* input, float* output);
void save_weights_to_nvs(const TrainableLayer* layer);
void load_weights_from_nvs(TrainableLayer* layer);

extern ExperienceReplay global_replay_buffer;
extern TrainableLayer global_trainable_layer;

#ifdef __cplusplus
}
#endif

#endif // INCREMENTAL_LEARNING_H
