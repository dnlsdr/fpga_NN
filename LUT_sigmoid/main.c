#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define LUT_SIZE 121
#define LUT_MIN (-6.0f)
#define LUT_MAX 6.0f
#define LUT_STEP 0.1f

#define MATRIX_ROWS 256
#define MATRIX_COLS 256
#define TEST_ITERS (MATRIX_ROWS * MATRIX_COLS)

float sigmoid_lut[LUT_SIZE];
float input_matrix[MATRIX_ROWS][MATRIX_COLS];
float output_lut[MATRIX_ROWS][MATRIX_COLS];
float output_exact[MATRIX_ROWS][MATRIX_COLS];

// LUT inicializálása
void init_sigmoid_lut() {
    for (int i = 0; i < LUT_SIZE; i++) {
        float x = LUT_MIN + i * LUT_STEP;
        sigmoid_lut[i] = 1.0f / (1.0f + expf(-x));
    }
}

// Véletlen input mátrix [-6, +6] tartományban
void generate_input_matrix() {
    for (int i = 0; i < MATRIX_ROWS; i++) {
        for (int j = 0; j < MATRIX_COLS; j++) {
            //lebegőpontos szám 0.0 és 1.0 között * 12 - 6
            float val = ((float)rand() / RAND_MAX) * 12.0f - 6.0f;
            input_matrix[i][j] = val;
        }
    }
}

float sigmoid_from_lut(float x) {
    if (x < LUT_MIN) x = LUT_MIN;
    if (x > LUT_MAX) x = LUT_MAX;
    int index = (int)((x - LUT_MIN) / LUT_STEP);
    return sigmoid_lut[index];
}

float sigmoid_exact(float x) {
    return 1.0f / (1.0f + expf(-x));
}

int main() {
    srand((unsigned int)time(nullptr));
    init_sigmoid_lut();
    generate_input_matrix();

    // --- LUT sigmoid mátrix kiszámítása ---
    //program indulása óta eltelt processzor idő tikkekben(egységekben)
    clock_t start = clock();
    for (int i = 0; i < MATRIX_ROWS; i++) {
        for (int j = 0; j < MATRIX_COLS; j++) {
            output_lut[i][j] = sigmoid_from_lut(input_matrix[i][j]);
        }
    }
    clock_t end = clock();
    //CLOCKS_PER_SEC másodpercenkénti tick, ami 1 millió
    double time_lut = (double)(end - start) / CLOCKS_PER_SEC;

    // --- Pontos sigmoid kiszámítása ---
    start = clock();
    for (int i = 0; i < MATRIX_ROWS; i++) {
        for (int j = 0; j < MATRIX_COLS; j++) {
            output_exact[i][j] = sigmoid_exact(input_matrix[i][j]);
        }
    }
    end = clock();
    double time_exact = (double)(end - start) / CLOCKS_PER_SEC;

    // --- Hibák elemzése ---
    float max_error = 0.0f;
    float sum_error = 0.0f;

    for (int i = 0; i < MATRIX_ROWS; i++) {
        for (int j = 0; j < MATRIX_COLS; j++) {
            float diff = fabsf(output_lut[i][j] - output_exact[i][j]);
            sum_error += diff;
            if (diff > max_error) max_error = diff;
        }
    }

    float avg_error = sum_error / TEST_ITERS;

    // --- Eredmények kiírása ---
    printf("Matrix size: %dx%d (total %d elements)\n", MATRIX_ROWS, MATRIX_COLS, TEST_ITERS);
    printf("LUT time:      %.6f sec\n", time_lut);
    printf("Function time:   %.6f sec\n", time_exact);
    printf("Max abs. error:        %.6f\n", max_error);
    printf("Average abs. error:    %.6f\n", avg_error);

    return 0;
}
