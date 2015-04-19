#include <stdio.h>
#include <omp.h>

//int main()
//{
//    int i;
//    int threadID = 0;
//    #pragma omp parallel for private(i, threadID)
//    for(i = 0; i < 16; i++ )
//    {
//        threadID = omp_get_thread_num();
//        #pragma omp critical
//        {
//            printf("Thread %d reporting\n", threadID);
//        }
//    }
//    return 0;
//}

void magic() {
#pragma omp parallel for
    for (int i = 0; i < 2; i++) {
        int threadID = omp_get_thread_num();
#pragma omp critical
        {
            printf("Thread %d reporting\n", threadID);
        }
    }

}

int main() {
    omp_set_nested(1);

#pragma omp parallel
#pragma omp single
    for (int i = 0; i < 5; i++) {
        magic();
    }
    return 0;
}
