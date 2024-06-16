#include <iostream>
#include <mpi.h>
#include <chrono>
#include <fstream>

int ROWS;
int COLS;
int ITERATIONS;

void updateCells(int **grid, int **nextGrid, int startRow, int endRow) {
    for (int i = startRow; i <= endRow; i++) {
        for (int j = 0; j < COLS; j++) {
            int neighbors = 0;
            for (int x = std::max(0, i-1); x <= std::min(ROWS-1, i+1); x++) {
                for (int y = std::max(0, j-1); y <= std::min(COLS-1, j+1); y++) {
                    if (x != i || y != j) {
                        neighbors += grid[x][y];
                    }
                }
            }

            if (grid[i][j] == 1 && neighbors < 2) {
                nextGrid[i][j] = 0;
            } else if (grid[i][j] == 1 && (neighbors == 2 || neighbors == 3)) {
                nextGrid[i][j] = 1;
            } else if (grid[i][j] == 1 && neighbors > 3) {
                nextGrid[i][j] = 0;
            } else if (grid[i][j] == 0 && neighbors == 3) {
                nextGrid[i][j] = 1;
            } else {
                nextGrid[i][j] = 0;
            }
        }
    }
}

void exchangeBoundaryRows(int **grid, int rank, int size, int startRow, int endRow) {

    int topProccess = (rank + size - 1) % size;
    int bottomProcess = (rank + 1) % size;

    int rowAboveMine = (startRow + ROWS - 1) % ROWS;
    int rowAfterMine = (endRow + 1) % ROWS;

    MPI_Status status;


    if (rank % 2 == 0) {
        MPI_Sendrecv(grid[startRow], COLS, MPI_INT, topProccess, rank, grid[rowAboveMine], COLS, MPI_INT, topProccess, topProccess, MPI_COMM_WORLD, &status);

        MPI_Sendrecv(grid[endRow], COLS, MPI_INT, bottomProcess, rank, grid[rowAfterMine], COLS, MPI_INT, bottomProcess, bottomProcess, MPI_COMM_WORLD, &status);
    }else{
        MPI_Sendrecv(grid[endRow], COLS, MPI_INT, bottomProcess, rank, grid[rowAfterMine], COLS, MPI_INT, bottomProcess, bottomProcess, MPI_COMM_WORLD, &status);

        MPI_Sendrecv(grid[startRow], COLS, MPI_INT, topProccess, rank, grid[rowAboveMine], COLS, MPI_INT, topProccess, topProccess, MPI_COMM_WORLD, &status);

    }
}


int main(int argc, char** argv) {
    std::ifstream input("input.txt");

    int randSeed;

    if(input.is_open()) {
        input >> randSeed >> COLS >> ROWS >> ITERATIONS;
        input.close();
    }

    auto startTime = std::chrono::system_clock::now();
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::cout << "Rank" << rank << std::endl;
    srand(randSeed);

    int **grid = new int*[ROWS];
    int **nextGrid = new int*[ROWS];
    for (int i = 0; i < ROWS; i++) {
        grid[i] = new int[COLS];
        nextGrid[i] = new int[COLS];
        for (int j = 0; j < COLS; j++) {
            grid[i][j] = rand() % 2;
            nextGrid[i][j] = grid[i][j];
        }
    }

    int startRow = rank * (ROWS / size);
    int endRow = (rank + 1) * (ROWS / size) - 1;

    for (int iter = 0; iter < ITERATIONS; iter++) {
        updateCells(grid, nextGrid, startRow, endRow);

        exchangeBoundaryRows(grid, rank, size, startRow, endRow);

        int **temp = grid;
        grid = nextGrid;
        nextGrid = temp;
    }

    MPI_Status status;

    if (rank == 0) {
        for (int i = 1; i < size; i++){
            for (int j = 0; j < ROWS / size; j++){
                int row = i * ROWS / size + j;
                MPI_Recv(grid[row], COLS, MPI_INT, i, row, MPI_COMM_WORLD, &status);
            }
        }

        std::ofstream output("output.txt");
        for(int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                output << grid[i][j] << " ";
            }
            output << std::endl;
        }
        output.close();
    }
    else{
        for (int i = 0; i < ROWS / size; i++){
            int row = startRow + i;
            MPI_Send(grid[row], COLS, MPI_INT, 0, row, MPI_COMM_WORLD);
        }
    }


    auto endTime = std::chrono::system_clock::now();

    std::cout << "Elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    MPI_Finalize();

    return 0;
}