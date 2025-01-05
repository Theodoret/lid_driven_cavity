#include <iostream>
#include <vector>
#include <math.h>
#include <time.h>
#include "header.hpp"

int main(){
    // Hyperparameters
    const int Nx = 51, Ny = 51, Nt = 10000;
    float lid = 1, Re = 1000,
        xmin = 0, xmax = 1, dx = (xmax - xmin)/(Nx - 1), 
        ymin = 0, ymax = 1, dy = (ymax - ymin)/(Ny - 1), 
        tmin = 0, tmax = 100, dt = (tmax - tmin)/Nt;
    float Nmax_SOR_psi = 200, psi_Toll = 0.01;

    // domain matrix
    std::vector<float> x, y;
    for (int i = 0; i<Nx; i++){
        x.push_back(xmin + dx*i);
    }
    for (int i = 0; i<Ny; i++){
        y.push_back(ymin + dy*i);
    }
    float X[Nx][Ny], Y[Nx][Ny];
    for (int i = 0; i<Nx; i++){
        for (int j = 0; j<Ny; j++){
            X[i][j] = x[i];
            Y[i][j] = y[j];
        }
    }

    // Parameters
    float psi[Nx][Ny] = {}, omega[Nx][Ny] = {}, u[Nx][Ny] = {}, v[Nx][Ny] = {};

    // imposing boundary
    for (int i = 0; i<Nx; i++){
        u[i][Ny] = lid;
    }

    // temporary parameters
    float omegax[Nx][Ny], omegay[Nx][Ny], omegaxx[Nx][Ny], omegayy[Nx][Ny], LHS, RHS;
    float psi_new[Nx][Ny], psi_SOR[Nx][Ny], psi_SORerr;

    // Time looping
    clock_t start = clock();
    for (int k = 0; k<Nt; k++){
        // imposing boundary on omega
        for (int i = 1; i<Nx - 1; i++){
            omega[i][0] = -2*psi[i][1]/pow(dy,2);
            omega[i][Ny-1] = -2*psi[i][Ny-2]/pow(dy,2) - lid*2/dx;
        }
        for (int j = 1; j<Ny - 1; j++){
            omega[0][j] = -2*psi[1][j]/pow(dx,2);
            omega[Nx-1][j] = -2*psi[Nx-2][j]/pow(dx,2);
        }

        // omega derivative with finite difference
        for (int i = 1; i<Nx - 1; i++){
            for (int j = 1; j<Ny - 1; j++){
                omegax[i][j] = (omega[i+1][j] - omega[i-1][j])/(2*dx);
                omegay[i][j] = (omega[i][j+1] - omega[i][j-1])/(2*dy); 
                
                omegaxx[i][j] = (omega[i+1][j] - 2*omega[i][j] + omega[i-1][j])/pow(dx,2);
                omegayy[i][j] = (omega[i][j+1] - 2*omega[i][j] + omega[i][j-1])/pow(dy,2);
            }
        }
        

        // vorticity function
        for (int i = 1; i<Nx - 1; i++){
            for (int j = 1; j<Ny - 1; j++){
                LHS = u[i][j]*omegax[i][j] + v[i][j]*omegay[i][j];
                RHS = (1/Re)*(omegaxx[i][j] + omegayy[i][j]);
                omega[i][j] = omega[i][j] + dt*(RHS - LHS);

                psi_new[i][j] = psi[i][j];
            }
        }

        // stream function
        for (int l = 0; l<Nmax_SOR_psi; l++){
            for (int i = 1; i<Nx - 1; i++){
                for (int j = 1; j<Ny - 1; j++){
                    psi_SOR[i][j] = 0.25*(psi_new[i+1][j] + psi_new[i-1][j] + psi_new[i][j+1] + psi_new[i][j-1] + dx*dy*omega[i][j]);
                }
            }
            psi_SORerr = 0;
            for (int i = 0; i<Nx; i++){
                for (int j = 0; j<Ny; j++){
                    psi_new[i][j] = psi_SOR[i][j];
                    psi_SORerr += abs(psi_new[i][j] - psi[i][j]);
                }
            }
            if (psi_SORerr <= psi_Toll){
                break;
            }
        }

        // velocity
        for (int i = 1; i<Nx - 1; i++){
            for (int j = 1; j<Ny - 1; j++){
                psi[i][j] = psi_new[i][j];

                u[i][j] = (psi[i][j+1] - psi[i][j-1])/(2*dy); // u = psiy
                v[i][j] = -(psi[i+1][j] - psi[i-1][j])/(2*dx); // u = -psix
            }
        }
    }
    clock_t end = clock();
    float runtime = end - start;

    // save the result in text
    write("X", *X, Nx, Ny);
    write("Y", *Y, Nx, Ny);
    write("psi", *psi, Nx, Ny);
    write("omega", *omega, Nx, Ny);
    write("u", *u, Nx, Ny);
    write("v", *v, Nx, Ny);

    std::cout << "runtime: " << runtime << " ms \n";
    return 0;
}