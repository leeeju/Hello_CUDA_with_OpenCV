#include <stdio.h>
 
int main()
{
    int InputData[5] = {1, 2, 3, 4, 5};
    int OutputData[5] = {0};
    int size;
 
    int* GraphicsCard_memory;
    size = sizeof(int);
 
    //그래픽카드 메모리의 할당
    cudaMalloc((void**)&GraphicsCard_memory, 5*size);
 
    //PC에서 그래픽 카드로 데이터 복사
    cudaMemcpy(GraphicsCard_memory, InputData, 5*size, cudaMemcpyHostToDevice);
 
    //그래픽 카드에서 PC로 데이터 복사
    cudaMemcpy(OutputData, GraphicsCard_memory, 5*size, cudaMemcpyDeviceToHost);
 
    //결과 출력
    for( int i = 0; i < 5; i++)
    {
        printf(" OutputData[%d] : %d\n", i, OutputData[i]);
    }
 
    //그래픽 카드 메모리의 해체
    cudaFree(GraphicsCard_memory);
 
    return 0;
}
