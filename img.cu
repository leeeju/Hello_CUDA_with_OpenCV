#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <Windows.h>

int main()
{
    //이미지의 정보를 읽기 위한 구조체
	BITMAPFILEHEADER hf;
	BITMAPINFOHEADER hinfo;
	RGBQUAD hRGB[256];
	FILE* fp;
	fp = fopen("cat.jpg", "rb");
	if (fp == NULL)
		return -1;
	fread(&hf, sizeof(BITMAPFILEHEADER), 1, fp); //비트맵 파일 정보 읽기
	fread(&hinfo, sizeof(BITMAPINFOHEADER), 1, fp); //비트맵 영상 정보 읽기
	fread(hRGB, sizeof(RGBQUAD), 256, fp);
	int imgSize = hinfo.biWidth * hinfo.biHeight; //이미지의 전체 바이트 길이 계산

    //이미지를 읽어 저장할 버퍼 할당
	BYTE* image = (BYTE*)malloc(imgSize);
	BYTE* output = (BYTE*)malloc(imgSize);
    //이미지 읽기
	fread(image, sizeof(BYTE), imgSize, fp);
	fclose(fp);

	//연산 처리
    addWithCuda(image, output, imgSize, -40);

	fp = fopen("o.bmp", "wb");
	fwrite(&hf, sizeof(BITMAPFILEHEADER), 1, fp);
	fwrite(&hinfo, sizeof(BITMAPINFOHEADER), 1, fp);
	fwrite(hRGB, sizeof(RGBQUAD), 256, fp);
	fwrite(output, sizeof(BYTE), imgSize, fp);
	fclose(fp);

    cudaError cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
