gcc -fopenmp -O3 trimm_main.c trimm_ijk_seq.c trimm_ijk_par.c -o trimm_ijk
gcc -fopenmp -O3 trimm_main.c trimm_kij_seq.c trimm_kij_par.c -o trimm_kij
