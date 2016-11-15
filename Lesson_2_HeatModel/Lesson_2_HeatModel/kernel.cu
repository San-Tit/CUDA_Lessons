
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_functions.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <fstream>
#include <cmath>
#define PI 3.141592653589793

// Время интегрирования
#define SIM_TIME 3000

//Шаг интегрирования;
#define D_T 0.1
#define iD_T 10

//Размерность сетки
#define DIM_X 10
#define DIM_Y 10
#define DIM_Z 10

//Граничные условия
#define S1 1
#define S2 1
#define S3 1
#define S4 0
#define S5 0
#define S6 0




cudaError_t computeWithCuda(double *dot);

__global__ void compute(double *dev_Q, double *dev_t, double *dev_T, double *dev_dT, double *dev_dot, double *dev_dott)
{
	//Условия задачи
	const int NVar = 7;
	const double Lx = (0.1*NVar);
	const double Ly = (0.2*NVar);
	const double Lz = (0.04*NVar);
	const double z1 = 0.6*Lz;
	const double z2 = 0.7*Lz;
	const double Zstar = 0.8*Lz;
	const double a1 = (0.00001 * NVar);
	const double a2 = 0.0002 * NVar;
	const double a3 = 5 * a1;
	//lambda->Lm;
	const double Lm1 = 0.02 * NVar;
	const double Lm3 = 0.02 * NVar;
	const double Lm2 = 5 * NVar;

	//число точек дискретизации
	const double dx = (Lx / (DIM_X - 1));
	const double dy = (Ly / (DIM_Y - 1));
	const double dz = (Lz / (DIM_Z - 1));

	int nu = threadIdx.x;
	int gamma = threadIdx.y;
	int dzeta = threadIdx.z;
	double t = 0;
	double A = 1000;
	while (t < 3000) {
		//расчет частных производных
		if (nu > 0 && nu < (DIM_X - 1) &&
			gamma > 0 && gamma < (DIM_Y - 1) &&
			dzeta > 0 && dzeta < (DIM_Z - 1)) {
			double dtdx = (*(dev_T + (nu - 1)*DIM_Y*DIM_Z + gamma*DIM_Z + dzeta) - 2 * *(dev_T + nu*DIM_Y*DIM_Z + gamma*DIM_Z + dzeta) + *(dev_T + (nu + 1)*DIM_Y*DIM_Z + gamma*DIM_Z + dzeta)) / (dx*dx);
			double dtdy = (*(dev_T + nu*DIM_Y*DIM_Z + (gamma - 1)*DIM_Z + dzeta) - 2 * *(dev_T + nu*DIM_Y*DIM_Z + gamma*DIM_Z + dzeta) + *(dev_T + nu*DIM_Y*DIM_Z + (gamma + 1)*DIM_Z + dzeta)) / (dy*dy);
			double dtdz = (*(dev_T + nu*DIM_Y*DIM_Z + gamma*DIM_Z + (dzeta - 1)) - 2 * *(dev_T + nu*DIM_Y*DIM_Z + gamma*DIM_Z + dzeta) + *(dev_T + nu*DIM_Y*DIM_Z + gamma*DIM_Z + (dzeta + 1))) / (dz*dz);
			*(dev_dT + nu*DIM_Y*DIM_Z + gamma*DIM_Z + dzeta) = a1*(dtdx + dtdy + dtdz)*D_T;
		}
		*(dev_T + nu*DIM_Y*DIM_Z + gamma*DIM_Z + dzeta) += *(dev_dT + nu*DIM_Y*DIM_Z + gamma*DIM_Z + dzeta);
		__syncthreads();
		//Синзронизация барьером

		//Граничные условия
		//S1 и S6 xy

		if (nu > 0 && nu < DIM_X &&
			gamma > 0 && gamma < DIM_Y) {
			*(dev_T + nu*DIM_Y*DIM_Z + gamma*DIM_Z + 0) = S1 * *(dev_T + nu*DIM_Y*DIM_Z + gamma*DIM_Z + 1) + A * (dz / Lm1)* *(dev_Q + nu*DIM_Y + gamma);
			*(dev_T + nu*DIM_Y*DIM_Z + gamma*DIM_Z + DIM_Z - 1) = S6 * *(dev_T + nu*DIM_Y*DIM_Z + gamma*DIM_Z + DIM_Z - 2);
		}

		//S2 и S4 xz
		if (nu > 0 && nu < DIM_X &&
			dzeta > 0 && dzeta < DIM_Z) {
			*(dev_T + nu*DIM_Y*DIM_Z + 0 * DIM_Z + dzeta) = S2 * *(dev_T + nu*DIM_Y*DIM_Z + 1 * DIM_Z + dzeta);
			*(dev_T + nu*DIM_Y*DIM_Z + (DIM_Z - 1) * DIM_Z + dzeta) = S4 * *(dev_T + nu*DIM_Y*DIM_Z + (DIM_Z - 2) * DIM_Z + dzeta);
		}

		//S3 и S5 yz

		if (gamma > 0 && gamma < DIM_X &&
			dzeta > 0 && dzeta < DIM_Z) {
			*(dev_T + 0 * DIM_Y*DIM_Z + gamma*DIM_Z + dzeta) = S2 * *(dev_T + 1 * DIM_Y*DIM_Z + gamma*DIM_Z + dzeta);
			*(dev_T + (DIM_X - 1) * DIM_Y*DIM_Z + gamma*DIM_Z + dzeta) = S2 * *(dev_T + (DIM_X - 2) * DIM_Y*DIM_Z + gamma*DIM_Z + dzeta);
		}
		if (nu == 5 && gamma == 5 && dzeta == 7) {
			*(dev_dot + int(t * iD_T)) = *(dev_T + 5 * DIM_Y*DIM_Z + 5 * DIM_Z + 7);
		}
		t += D_T;
		__syncthreads();
	}
}

int main()
{
	double dot[int(SIM_TIME / D_T)];
	// Add vectors in parallel.
	cudaError_t cudaStatus = computeWithCuda(dot);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	std::ofstream fout;
	fout.open("sim.txt");
	for (int i = 0; i < int(SIM_TIME * iD_T); i++) {
		fout << dot[i] << '\n';
	}
	fout.close();


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	system("pause");
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t computeWithCuda(double *dot)
{
	double dT[DIM_X][DIM_Y][DIM_Z];
	double T[DIM_X][DIM_Y][DIM_Z];
	double Q[DIM_X][DIM_Y];
	for (int i = 0; i < DIM_X; i++)
		for (int j = 0; j < DIM_Y; j++)
			for (int z = 0; z < DIM_Z; z++) {
				dT[i][j][z] = 0;
				T[i][j][z] = 0;
			}
	int i = 3;
	double psi1 = PI*i / DIM_X;
	double psi2 = PI*i / DIM_Y;
	for (int x = 0; x < 10; x++)
		for (int y = 0; y < 10; y++)
			Q[x][y] = (cos(psi1*(x + 1) / 2))*(cos(psi2*(y + 1) / 2));

	double *dev_Q = 0;
	double *dev_t = 0;
	double *dev_T = 0;
	double *dev_dT = 0;
	double *dev_dot = 0;
	double *dev_dott = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Выделение памяти для массива мод воздейстивя
	cudaStatus = cudaMalloc((void**)&dev_Q, sizeof(double) * DIM_X * DIM_Y);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Выделение памяти для массива состояния
	cudaStatus = cudaMalloc((void**)&dev_T, sizeof(double) * DIM_X * DIM_Y * DIM_Z);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Выделение памяти для массива изменений
	cudaStatus = cudaMalloc((void**)&dev_dT, sizeof(double) * DIM_X * DIM_Y * DIM_Z);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Выделение памяти для массива временной характеристики точки
	cudaStatus = cudaMalloc((void**)&dev_dot, sizeof(double)*SIM_TIME * iD_T);
	fprintf(stderr, "%d \n", SIM_TIME * iD_T);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Выделение памяти для времени
	cudaStatus = cudaMalloc((void**)&dev_t, sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Выделение памяти для 
	cudaStatus = cudaMalloc((void**)&dev_t, sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Копирование массива воздействий.
	cudaStatus = cudaMemcpy(dev_Q, Q, sizeof(double) * DIM_X * DIM_Y, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Копирование массива состояний.
	cudaStatus = cudaMemcpy(dev_T, T, sizeof(double) * DIM_X * DIM_Y * DIM_Z, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Копирование массива изменений
	cudaStatus = cudaMemcpy(dev_dT, dT, sizeof(double) * DIM_X * DIM_Y * DIM_Z, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	dim3 threadsPerBlock(DIM_X, DIM_Y, DIM_Z);
	compute << < 1, threadsPerBlock >> > (dev_Q, dev_t, dev_T, dev_dT, dev_dot, dev_dott);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(dot, dev_dot, int(SIM_TIME * iD_T) * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_t);
	cudaFree(dev_T);
	cudaFree(dev_dT);
	cudaFree(dev_dot);
	cudaFree(dev_dott);
	return cudaStatus;
}
