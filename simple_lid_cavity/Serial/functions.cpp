#include "header.hpp"
#include <string>
#include <fstream>

// Axis::Axis(int min, int max, int n, float h) : min(min), max(max), n(n), h(h) {}
// float Axis::get_h(){
//     return h;
// }

// float finite_difference(float h, float dx, float dy, int Nx, int Ny){
//     float hx[Nx][Ny], hy[Nx][Ny], hxx[Nx][Ny], hyy[Nx][Ny];

//     for (int i = 0; i<Nx; i++){
//         for (int j = 0; i<Ny; j++){
//             hx[i][j] = h[i][j];
//         }
//     }
// }

void write(std::string name, float *matrix, int Nx, int Ny){
    std::string extension = ".txt";
    std::ofstream file_text(name + extension);
    for (int i = 0; i<Nx; i++){
        for (int j = 0; j<Ny; j++){
            if ( j != 0){
                file_text << ",";
            }
            file_text << matrix[j*Nx + i];
        }
        file_text << "\n";
    }
}