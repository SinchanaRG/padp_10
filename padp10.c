#include <sys/time.h>
#include <stdio.h>
#define SIZE 4000
double a[SIZE][SIZE];
double b[SIZE][SIZE];
double c[SIZE][SIZE];
double d[SIZE][SIZE];
int main()
{
int i,j,k;
struct timevaltim;
double t1, t2;
doubletmp;
// Initialize matrices.
for (i = 0; i<SIZE; ++i) {
for (j = 0; j<SIZE; ++j) {
a[i][j] = (double)(i + j);
b[i][j] = (double)(i - j);
c[i][j] = 0.0f;
d[i][j] = 0.0f;
}
}
// Time stamp t1
gettimeofday(&tim, NULL);
t1=tim.tv_sec+(tim.tv_usec/1000000.0);
// Compute matrix multiplication.
#pragma acc data copyin(a,b) copy(c)
#pragma acc kernels
#pragma acc loop tile(32,32)
for (i = 0; i &lt; SIZE; ++i) {
for (j = 0; j &lt; SIZE; ++j) {
tmp=0.0f;
#pragma acc loop reduction(+:tmp)
for (k = 0; k<SIZE; ++k) {
tmp += a[i][k] * b[k][j];
}
c[i][j] = tmp;
}
}
// Time stamp t2, elapsed time OpenACC
gettimeofday(&tim, NULL);
t2=tim.tv_sec+(tim.tv_usec/1000000.0);
printf("%.6lf seconds with OpenACC \n", t2-t1);

// Check the OpenACC result matrix
for (i = 0; i<SIZE; ++i)
for (j = 0; j<SIZE; ++j)
if(c[i][j] != d[i][j]) {
printf("Error %d %d %f %f \n", i,j, c[i][j], d[i][j]);
exit(1);
}
printf("OpenACC matrix multiplication test was successful!\n");
return 0;
}
