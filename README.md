# fastjet_gpu
Porting [FastJet](http://fastjet.fr/) software package to CUDA.

Using FastJet 3.3.2

# Compile 
```
make
```

# Run
```
./[grid|n_array|tri_matrix] [OPTION] < [FILE] 
```
or 
```
./[grid|n_array|tri_matrix] [OPTION] -file [FILE]
```
Example
```
./tri_matrix -antikt -r 0.4 -p 1.0 < hltAK4CaloJets.dat
```

# Options
`--ptmin | -p` minimum pT default `0.0`.

`-r` | `-R` clustering radius default `1.0`.

`--repeat` | `-repeat` repeat execution n times, default `1`.

`--sort` | `-s` sort the sort the jets by decreasing pT default `true`.

`--cartesian` output px, py, pz, E

`--polar` output eta, phi, pT

`--kt` | `-kt`

`--anti-kt` | `-antikt`

`--cambridge-aachen` | `-cam`

`--file` | `-f` input file name.

`--csv` | `-csv` output timing data in csv format.
