#ifndef C_SV_H
#define C_SV_H

#include "tenCor.h"

#define MAX_CHARS_PER_LINE 4096

typedef struct {
    int rows; // not including header
    int cols; // not including header
    char **headers;
    tensor **elements;
} dataTable;

dataTable *readCSVToDataTable(FILE *src);
void destroyDataTable(dataTable *dt);
void cleanDataTable(dataTable *data);
bool splitDataTable2(dataTable *inputData, int extractCol, tensor tenX, tensor vecY);

#endif // C_SV_H