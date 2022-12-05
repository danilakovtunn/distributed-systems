/* Include benchmark-specific header. */
#include "jacobi-1d.h"
#include <mpi.h>
double bench_t_start, bench_t_end;
int size, rank;

static
double rtclock() {
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, NULL);
    if (stat != 0)
        printf ("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}


void bench_timer_start() {
    bench_t_start = rtclock ();
}


void bench_timer_stop() {
    bench_t_end = rtclock ();
}


void bench_timer_print() {
    printf ("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
}


static
void init_array (int n, double *A, double *B) {
    int i;

    for (i = 0; i < n; i++) {
        A[i] = ((double) i + 2) / n;
        B[i] = ((double) i + 3) / n;
    }
}


static
void print_array(int n, double *A) {
    int i;

    fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
    fprintf(stderr, "begin dump: %s", "A");
    for (i = 0; i < n; i++) {
        if (i % 20 == 0) 
            fprintf(stderr, "\n");
        fprintf(stderr, "%0.2lf ", A[i]);
    }
    fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}


static
void kernel_jacobi_1d(int tsteps, int n, double *A, double *B) {
    int t, i, count, ibeg, iend, right_rank, left_rank;
    MPI_Status status;
    MPI_Request req;
    right_rank = 0;
    left_rank = 0;
    count = (n - 3) / size + 1;
    ibeg = rank * count + 1;
    iend = (rank + 1) * count;
    // Если нитей больше чем данных, которые необходимо обработать
    if (ibeg >= n - 2) {
        ibeg = -1;
        iend = -1;
    }
    // У последней нити, которая попадает в нужный нам диапозон меняем iend на нужный нам и 
    // сообщаем какой у нити ранг процессу size - 1
    if (iend >= n - 3) {
        iend = n - 3;
        right_rank = size - 1;
        MPI_Isend(&rank, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD, &req);
    }
    // У нити size - 1 корректируем ibeg & iend и получаем номер последней нормальной нити
    if (rank == size - 1) { 
        printf("hehe\n");
        MPI_Recv(&left_rank, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        printf("hehe\n");
        if (ibeg == -1)
            ibeg = n - 2;
        iend = n - 2; 
        //Случай, когда нормально разделилось без постороннего вмешательства
        if (left_rank == right_rank) 
            left_rank--;
    }
    
    for (t = 0; t < tsteps; t++) {
        if (ibeg == -1)
            continue;
        for (i = ibeg; i <= iend; i++)
            B[i] = 0.33333 * (A[i - 1] + A[i] + A[i + 1]);
        
        if (rank == 0) {
            if (size != 1) {
                MPI_Send(&B[iend], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
                MPI_Recv(&B[iend + 1], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &status);
            }
        } 
        // Кидаем данные левой нити
        else if (rank == size - 1) {
            MPI_Send(&B[ibeg], 1, MPI_DOUBLE, left_rank, 0, MPI_COMM_WORLD);
            MPI_Recv(&B[ibeg - 1], 1, MPI_DOUBLE, left_rank, 0, MPI_COMM_WORLD, &status);
        }
        // кидаем данные нити size - 1
        else if (right_rank) {
            MPI_Send(&B[iend], 1, MPI_DOUBLE, right_rank, 0, MPI_COMM_WORLD);
            MPI_Send(&B[ibeg], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&B[iend + 1], 1, MPI_DOUBLE, right_rank, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&B[ibeg - 1], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &status);
        }
        // Кидаем и получаем данные у соседних нитей
        else {
            MPI_Send(&B[iend], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
            MPI_Send(&B[ibeg], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&B[iend + 1], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&B[ibeg - 1], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &status);
        }

        // Аналогично для матрицы A
        for (i = ibeg; i <= iend; i++)
            A[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
        
        if (rank == 0) {
            if (size != 1) {
                MPI_Send(&A[iend], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
                MPI_Recv(&A[iend + 1], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &status);
            }
        } 
        else if (rank == size - 1) {
            MPI_Send(&A[ibeg], 1, MPI_DOUBLE, left_rank, 0, MPI_COMM_WORLD);
            MPI_Recv(&A[ibeg - 1], 1, MPI_DOUBLE, left_rank, 0, MPI_COMM_WORLD, &status);
        }
        else if (right_rank) { // != 0
            MPI_Send(&A[iend], 1, MPI_DOUBLE, right_rank, 0, MPI_COMM_WORLD);
            MPI_Send(&A[ibeg], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&A[iend + 1], 1, MPI_DOUBLE, right_rank, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&A[ibeg - 1], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &status);
        }
        else {
            MPI_Send(&A[iend], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
            MPI_Send(&A[ibeg], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&A[iend + 1], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&A[ibeg - 1], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &status);
        }
    }
    // Собираем новые данные на нити size - 1
    if (ibeg != -1 && size != 1) {
        if (rank != size - 1) {
            MPI_Send(&B[ibeg], iend - ibeg + 1, MPI_DOUBLE, size - 1, 0, MPI_COMM_WORLD);
        }
        else {
            int i;
            for (i = 0; i <= left_rank; i++) {
                MPI_Recv(&B[i * count + 1], (i == left_rank) ? (n - 3 - i * count) : count, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
            
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (ibeg != -1 && size != 1) {
        if (rank != size - 1) {
            MPI_Send(&A[ibeg], iend - ibeg + 1, MPI_DOUBLE, size - 1, 0, MPI_COMM_WORLD);
            //printf("%d %d %d\n", rank, ibeg, ibeg + count - 1);
        }
        else {
            int i;
            for (i = 0; i <= left_rank; i++) {
                MPI_Recv(&A[i * count + 1], (i == left_rank) ? (n - 3 - i * count) : count, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
            
            }
        }
    }
}


int main(int argc, char *argv[]) {
    int n_s[] = {30, 120, 400, 2000, 4000, 8000, 16000, 32000, 64000};
    int step_s[] = {20, 40, 100, 500, 1000, 2000, 4000, 8000, 16000};
    int i;

    double *A; 
    double *B;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    for (i = 0; i < 1; i++) {
        int n = n_s[i];
        int tsteps = step_s[i];
        
        if (rank == size - 1) {
            printf("n=%d tsteps=%d threads=%d\n", n, tsteps, size);
        }

        A = (double *) malloc (n * sizeof(double));
        B = (double *) malloc (n * sizeof(double));

        init_array (n, A, B);
        
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == size - 1) {
            printf("Начато замеряться время\n");
            bench_timer_start();
        }
        
        kernel_jacobi_1d(tsteps, n, A, B);
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == size - 1) {
            bench_timer_stop();
            bench_timer_print();
            
            //printf("A\n");
            //print_array(n, A);
            printf("B\n");
            print_array(n, B);
        }
        
        free(A);
        free(B);
    }
    MPI_Finalize();
    return 0;
}
