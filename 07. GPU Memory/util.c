
float* get_buffer(int n_size) {
    float* buffer = (float*)malloc(n_size * sizeof(float));
    
    time_t t;
    srand((unsigned) time(&t));
    
    for (int i = 0; i < n_size; i++) {
        //buffer[i] = (float)rand()/(float)(RAND_MAX/100);
        buffer[i] = i;
    }
    return buffer;
}

void check_result(float *p_A, float *p_B, int n_size) {
    int compare = 0;
    for (int i = 0; i < n_size; i++) {
        compare += (p_A[i] != p_B[i]) ? 1 : 0;
    }
    printf("Result: %d\n", compare);
}