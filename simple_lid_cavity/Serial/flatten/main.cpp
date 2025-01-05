#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>

int main(){
    // Hyperparameters
    const int Nx = 51, Ny = 51, N = Nx*Ny, Nt = 10000;
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
    float X[N], Y[N];
    for (int i = 0; i<Nx; i++){
        for (int j = 0; j<Ny; j++){
            // X[j*Nx + i] = i;
            // Y[j*Nx + i] = j;
            X[j*Nx + i] = x[i];
            Y[j*Nx + i] = y[j];
        }
    }

    // Parameters
    float psi[N] = {}, omega[N] = {}, u[N] = {}, v[N] = {};

    // imposing boundary
    for (int i = 0; i<Nx; i++){
        u[(Ny-1)*Nx + i] = lid;
    }

    // temporary parameters
    float omegax[N], omegay[N], omegaxx[N], omegayy[N], LHS, RHS;
    float psi_new[N], psi_SOR[N], psi_SORerr;

    // Time looping
    for (int k = 0; k<Nt; k++){
        // imposing boundary on omega
        for (int i = 1; i<Nx - 1; i++){
            omega[i] = -2*psi[Nx + i]/pow(dy,2);
            omega[(Ny-1)*Nx + i] = -2*psi[(Ny-2)*Nx + i]/pow(dy,2) - lid*2/dx;
        }
        for (int j = 1; j<Ny - 1; j++){
            omega[j*Nx] = -2*psi[j*Nx + 1]/pow(dx,2);
            omega[j*Nx + Nx-1] = -2*psi[j*Nx + Nx-2]/pow(dx,2);
        }

        // omega derivative with finite difference
        for (int i = 1; i<Nx - 1; i++){
            for (int j = 1; j<Ny - 1; j++){
                omegax[j*Nx + i] = (omega[j*Nx + (i+1)] - omega[j*Nx + (i-1)])/(2*dx);
                omegay[j*Nx + i] = (omega[(j+1)*Nx + i] - omega[(j-1)*Nx + i])/(2*dy); 
                
                omegaxx[j*Nx + i] = (omega[j*Nx + (i+1)] - 2*omega[j*Nx + i] + omega[j*Nx + (i-1)])/pow(dx,2);
                omegayy[j*Nx + i] = (omega[(j+1)*Nx + i] - 2*omega[j*Nx + i] + omega[(j-1)*Nx + i])/pow(dy,2);
            }
        }
        

        // vorticity function
        for (int i = 1; i<Nx - 1; i++){
            for (int j = 1; j<Ny - 1; j++){
                LHS = u[j*Nx + i]*omegax[j*Nx + i] + v[j*Nx + i]*omegay[j*Nx + i];
                RHS = (1/Re)*(omegaxx[j*Nx + i] + omegayy[j*Nx + i]);
                omega[j*Nx + i] = omega[j*Nx + i] + dt*(RHS - LHS);

                psi_new[j*Nx + i] = psi[j*Nx + i];
            }
        }

        // stream function
        for (int l = 0; l<Nmax_SOR_psi; l++){
            for (int i = 1; i<Nx - 1; i++){
                for (int j = 1; j<Ny - 1; j++){
                    psi_SOR[j*Nx + i] = 0.25*(psi_new[j*Nx + (i+1)] + psi_new[j*Nx + (i-1)] + psi_new[(j+1)*Nx + i] + psi_new[(j-1)*Nx + i] + dx*dy*omega[j*Nx + i]);
                }
            }
            psi_SORerr = 0;
            for (int i = 0; i<Nx; i++){
                for (int j = 0; j<Ny; j++){
                    psi_new[j*Nx + i] = psi_SOR[j*Nx + i];
                    psi_SORerr += abs(psi_new[j*Nx + i] - psi[j*Nx + i]);
                }
            }
            if (psi_SORerr <= psi_Toll){
                break;
            }
        }

        // velocity
        for (int i = 1; i<Nx - 1; i++){
            for (int j = 1; j<Ny - 1; j++){
                psi[j*Nx + i] = psi_new[j*Nx + i];

                u[j*Nx + i] = (psi[(j+1)*Nx + i] - psi[(j-1)*Nx + i])/(2*dy); // u = psiy
                v[j*Nx + i] = -(psi[j*Nx + (i+1)] - psi[j*Nx + (i-1)])/(2*dx); // u = -psix
            }
        }
    }

    // save the result in text
    std::ofstream psi_text("psi.txt");
    for (int i = 0; i<Nx; i++){
        for (int j = 0; j<Ny; j++){
            if ( j != 0){
                psi_text << ",";
            }
            psi_text << psi[j*Nx + i];
        }
        psi_text << "\n";
    }

    std::cout << "Done";
    return 0;
}