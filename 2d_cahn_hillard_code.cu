//2d cahn hillard with initial condition as random noise using spectral with periodic boundary conditions
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include "device_launch_parameters.h"

#define sizex 512
#define sizey 512
#define dt 1.0e-3
#define PI 3.14159265358979323846
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32
#define NT 100000 //number of time steps

int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

__global__ void iterate1(cufftComplex *c, cufftComplex *g)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	int p = i + j * sizex;

	g[p].x = 2 * c[p].x*(1 - c[p].x)*(1 - 2 * c[p].x);
}

__global__ void iterate2(cufftComplex *c, cufftComplex *g, double *dkx, double *dky)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	double kx, ky;
	if (i < sizex / 2)
		kx = i * *dkx;
	else
		kx = (i - sizex) * *dkx;

	if (j < sizey / 2)
		ky = j * *dky;
	else
		ky = (j - sizey) * *dky;

	int p = i + j * sizex;

	c[p].y = 0;
	g[p].y = 0;

	c[p].x = (c[p].x - dt * (kx*kx + ky * ky)*g[p].x) / (1 + 2 * (kx*kx + ky*ky) * (kx*kx + ky*ky) * dt);
}

__global__ void iterate3(cufftComplex *c)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	int p = i + j * sizex;

	c[p].x = c[p].x/(sizex*sizey);
	c[p].y = 0;
}

int main()
{
	clock_t t;
	t = clock();

	cufftHandle plan;

	cufftComplex c[sizex][sizey], *gpu_c, *gpu_g;

	char output_filename[100];
	char str1[]=".//output//order_parameter_";
	char str2[10];

	double *dkx, *dky;
	double *gpu_dkx, *gpu_dky;

	double dkx0 = 2 * PI / sizex;
	double dky0 = 2 * PI / sizey;

	dkx = &dkx0;
	dky = &dky0;

	srand(time(0));

	double random;

	FILE *file0;
	file0 = fopen(".//output//order_parameter_000.vtk", "w");

	if (file0 == NULL)
	{
		printf("Can't open order_parameter_000.vtk file for writting\n");
	}

	fprintf(file0, "# vtk DataFile Version 3.0\n");
	fprintf(file0, "Order Parameter data\n");
	fprintf(file0, "ASCII\n");
	fprintf(file0, "DATASET STRUCTURED_POINTS\n");
	fprintf(file0, "DIMENSIONS %d %d 1\n", sizex, sizey);
	fprintf(file0, "ORIGIN 0 0 0\n");
	fprintf(file0, "SPACING 1 1 1\n");
	fprintf(file0, "POINT_DATA %d\n", sizex*sizey);
	fprintf(file0, "SCALARS order_parameter double\n");
	fprintf(file0, "LOOKUP_TABLE default\n");

	for (int i = 0; i < sizex; i++)
	{
		for (int j = 0; j < sizey; j++)
		{
			random = rand();
			random /= RAND_MAX;
			random = 0.25 + 0.5*  random;
			c[i][j].x = random;
			c[i][j].y = 0.0f;

			fprintf(file0, "%f\n", c[i][j].x);
		}
	}

	fclose(file0);

	cudaMalloc(&gpu_c, sizeof(cufftComplex)*sizex*sizey);//note that gpu_c is 1D variable
	cudaMalloc(&gpu_g, sizeof(cufftComplex)*sizex*sizey);
	cudaMalloc(&gpu_dkx, sizeof(double));
	cudaMalloc(&gpu_dky, sizeof(double));

	cudaMemcpy(gpu_c, c, sizeof(cufftComplex)*sizex*sizey, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_dkx, dkx, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_dky, dky, sizeof(double), cudaMemcpyHostToDevice);

	cufftPlan2d(&plan, sizex, sizey, CUFFT_C2C);

	dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 dimGrid(iDivUp(sizex, BLOCK_SIZE_X), iDivUp(sizey, BLOCK_SIZE_Y));

	for (int t = 1; t <= NT; t++)
	{
		iterate1<<< dimGrid, dimBlock >>>(gpu_c, gpu_g);

		cufftExecC2C(plan, gpu_c, gpu_c, CUFFT_FORWARD);
		cufftExecC2C(plan, gpu_g, gpu_g, CUFFT_FORWARD);

		iterate2<<< dimGrid, dimBlock >>>(gpu_c, gpu_g, gpu_dkx, gpu_dky);

		cufftExecC2C(plan, gpu_c, gpu_c, CUFFT_INVERSE);

		iterate3<<< dimGrid, dimBlock >>>(gpu_c);

		if(t%100 == 0)
		{		//print order parameter in vtk file
			cudaMemcpy(c, gpu_c, sizeof(cufftComplex)*sizex*sizey, cudaMemcpyDeviceToHost);

			sprintf(str2, "%03d.vtk", t);
			strcat(output_filename, str1);
			strcat(output_filename, str2);

			FILE *file0;
			file0 = fopen(output_filename, "w");

			if (file0 == NULL)
			{
				printf("Can't open order_parameter_%03d.vtk file for writting\n", t);
			}

			fprintf(file0, "# vtk DataFile Version 3.0\n");
			fprintf(file0, "Order Parameter data\n");
			fprintf(file0, "ASCII\n");
			fprintf(file0, "DATASET STRUCTURED_POINTS\n");
			fprintf(file0, "DIMENSIONS %d %d 1\n", sizex, sizey);
			fprintf(file0, "ORIGIN 0 0 0\n");
			fprintf(file0, "SPACING 1 1 1\n");
			fprintf(file0, "POINT_DATA %d\n", sizex*sizey);
			fprintf(file0, "SCALARS order_parameter double\n");
			fprintf(file0, "LOOKUP_TABLE default\n");
	
			for (int x = 0; x < sizex; x++)
			{
				for (int y = 0; y < sizey; y++)
				{
					fprintf(file0, "%lf\n", c[x][y].x);
				}
			}

			fclose(file0);
			//printf("closed order_parameter_%03d.vtk file\n", t);

			for(int k=0;k<strlen(output_filename);k++)
			{
    				output_filename[k] = 0;
			}
		}
		printf("%f%%\n", (double)t*100/NT);

	}

	cufftDestroy(plan);

	cudaFree(gpu_c);
	cudaFree(gpu_g);
	cudaFree(gpu_dkx);
	cudaFree(gpu_dky);

	t = clock() - t;
	double timetaken = ((double)t) / CLOCKS_PER_SEC;
	printf("time taken =%f\n", timetaken);

	return 0;
}
