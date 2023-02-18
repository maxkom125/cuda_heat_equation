#ifndef SEQHEATEQ_H
#define SEQHEATEQ_H

double Get_U(double* previous_row, long int i);
void Save_row(const double* current_row, const int rank, const long int array_len);
void Swap_double_arrays(double** array1, double** array2);
void SequentialHeatEquation(int coordSteps, int timeSteps, float leftValue, float rightValue, std::string testFile);

#endif //SEQHEATEQ_H