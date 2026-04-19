#include <stdio.h>
#include <omp.h>
int main(void) {
    int height = 10;
    int sum = 0;
    int y;
    #pragma omp parallel for reduction(+:sum)
    for(y = 0; y < height; y++) {
        sum += y;
    }
    printf("sum=%d\n", sum);
    return 0;
}
