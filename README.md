# fastjet_gpu
porting the fastjet software package to work on GPUS
private repo for now.

### Compile
```
nvcc file.cu -o file
```

### Run
```
./file < single-event.dat
```

### Enable Output and Benchmarking (Only for tri_matrix.cu)
Inside tri_matrix.cu
change the preprocessor directives in the beginning of the file.
```
#define OUTPUT_JETS false
#define BENCH true
```

### Multiple events (Only for tri_matrix.cu)
Change the `num_events` variable to the number of events needed.
```
 // Increase the number to process more events
int num_events = 1;

// Loop events
for (int event = 0; event < num_events; event++) {
```

