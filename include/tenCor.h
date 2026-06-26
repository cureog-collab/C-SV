#ifndef MAIN_H
#define MAIN_H

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define EPSILON 1E-10

typedef struct {
    double *data;
    int *shape;
    int dimensions;
    int size;
} tensor;

// =========================================================================================================
// MEMORY HANDLING 
// =========================================================================================================
tensor *createTensor(int dim, const int *shape);
tensor *createIdentityMatrix(int matSize);
tensor *copyTensor(const tensor *ten);
void destroyTensor(tensor *ten);

// =========================================================================================================
// TENSOR INITIALIZATION
// =========================================================================================================
void tensorFill(tensor *ten, double A);
void tensorFillZEROS(tensor *ten);
void tensorFillONES(tensor *ten);
void tensorRandomUniformUnity(tensor *ten); 
void tensorRandomNormal(tensor *ten, double mean, double sigma);

// =========================================================================================================
// ELEMENT-WISE MATHEMATICS & FORWARD PASS
// =========================================================================================================
void tensorScale(tensor *ten, double scalar);
tensor *tensorAdd(const tensor *ten1, const tensor *ten2, bool isAdd);
tensor *tensorAddBias(const tensor *ten, const tensor *bias);
tensor *tensorAddScalar(const tensor *ten, double scalar);
tensor *tensorHadamardProduct(const tensor *ten1, const tensor *ten2);
tensor *tensorDivide(const tensor *ten1, const tensor *ten2);

tensor *tensorRelu(const tensor *ten);
tensor *tensorSigmoid(const tensor *ten);
tensor *tensorTanh(const tensor *ten);
tensor *tensorSoftmax(const tensor *ten, int axis);

bool checkShapeSim(const tensor *ten1, const tensor *ten2);

// =========================================================================================================
// BACKPROPAGATION
// =========================================================================================================
tensor *tensorReluDerivative(const tensor *ten);
tensor *tensorSigmoidDerivative(const tensor *ten);
tensor *tensorTanhDerivative(const tensor *ten);

// =========================================================================================================
// LOSS FUNCTIONS
// =========================================================================================================
double tensorMSE(const tensor *predictions, const tensor *labels);
double tensorCrossEntropy(const tensor *predictions, const tensor *labels);

// =========================================================================================================
// LINEAR ALGEBRA CORE
// =========================================================================================================
tensor *tensorTranspose(const tensor *ten, const int *axes);
tensor *tensorInverse(const tensor *ten);
tensor *tensorMultiply(const tensor *ten1, const tensor *ten2, bool trans1, bool trans2);
double matrixDet(const tensor *mat);
void gaussianElimination(tensor *mat, int entryRowIdx, int entryColIdx, int *swapSign);

// =========================================================================================================
// REDUCTION & STATISTICS
// =========================================================================================================
double tensorSum(const tensor *ten);
double tensorMean(const tensor *ten);
tensor *tensorSumByAxis(const tensor *ten, int axis);
tensor *tensorMaxByAxis(const tensor *ten, int axis);
tensor *tensorArgmax(const tensor *ten, int axis);

// =========================================================================================================
// SHAPE MORPHING
// =========================================================================================================
tensor *tensorReshape(const tensor *ten, int *newShape, int newDim);
tensor *tensorFlatten(const tensor *ten);

// =========================================================================================================
// IN-PLACE COUNTERPARTS
// =========================================================================================================
bool tensorAddBiasInPlace(tensor *ten, const tensor *bias);
void tensorReluInPlace(tensor *ten);
void tensorSigmoidInPlace(tensor *ten);
void tensorTanhInPlace(tensor *ten);
bool tensorSoftmaxInPlace(tensor *ten, int axis);
void tensorReluDerivativeInPlace(tensor *ten);
void tensorSigmoidDerivativeInPlace(tensor *ten);
void tensorTanhDerivativeInPlace(tensor *ten);
bool tensorTransposeInPlace(tensor *ten, const int *axes);
bool tensorInverseInPlace(tensor *ten);
bool tensorReshapeInPlace(tensor *ten, int *newShape, int newDim);
bool tensorFlattenInPlace(tensor *ten);
bool tensorSumByAxisInPlace(tensor *ten, int axis);
bool tensorMaxByAxisInPlace(tensor *ten, int axis);
bool tensorArgmaxInPlace(tensor *ten, int axis);
bool tensorAddScalarInPlace(tensor *ten, double scalar);

#endif