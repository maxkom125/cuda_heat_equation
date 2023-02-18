#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstring>

#include "seqheateq.h"

const float integrStep = 0.3; //TODO: relocate
float GetU(float* previousRow, long int i) {
    return previousRow[i] + integrStep * (previousRow[i + 1] - 2 * previousRow[i] + previousRow[i - 1]);
}

void SwapFloatArrays(float** array1, float** array2){
    float* x = *(array1);
    *array1 = *array2;
    *array2 = x;
}

void SequentialHeatEquation(int coordSteps, int timeSteps, float leftValue, float rightValue, std::string testFile) {
    //Compute heat equation on CPU and compare results with testFile
    float* currentRow  = new float[coordSteps](); //array with zeros
    float* previousRow = new float[coordSteps](); //array with zeros
    
    previousRow[0] = leftValue;
    previousRow[coordSteps - 1] = rightValue;
    currentRow[0] = leftValue;
    currentRow[coordSteps - 1] = rightValue;

    for(long int n = 0; n < timeSteps; n++) {
        for (long int m = 1; m < coordSteps - 1; m++) {
            currentRow[m] = GetU(previousRow, m);
        }

        SwapFloatArrays(&previousRow, &currentRow);
    }
    long int m_i;
    
    // FILE* fp = fopen("Seq_Heat_out.txt", "w");

    // for(m_i = 0; m_i < coordSteps; m_i++)
    //     fprintf(fp, "%f\n", previousRow[m_i]); //h * (m_i),
    // fclose(fp);
    
    //Compare results with testFile
    std::string line;
    std::ifstream file(testFile);
    if (file.is_open()) {
        long int i = 0;
        int f = 0;
        float tolerance = 0.0001;
        while (getline(file, line)) {
            if (coordSteps <= i) {
                std::cout << "Different coordSteps!" << std::endl;
                f++;
                break;
            }

            if (fabs(previousRow[i] - stod(line)) > tolerance) {
                std::cout << i << ": CPU: " << previousRow[i] << " GPU: " << stof(line) << std::endl;
                f++;
            }
            i++;
        }
        if (f == 0)
            std::cout << "TEST SUCCESS" << std::endl;
        else
            std::cout << "TEST ERRORS: " << f << std::endl;

        file.close();
    } else {
        std::cout << "Unable to open test file" << std::endl;
    }

    delete[] previousRow;
    delete[] currentRow;
    return;
}
