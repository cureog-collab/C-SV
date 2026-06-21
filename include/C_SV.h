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

typedef enum {
    IMPUTE_MEAN = 1,
    IMPUTE_MEDIAN = 2,
    IMPUTE_MODE = 3,
    IMPUTE_ZERO = 4
} imputeStrategy;

typedef enum {
    SCALE_MINMAX = 1,
    SCALE_STANDARD = 2,
    SCALE_ROBUST = 3
} normalizeStrategy;

// ============================================================================
// MEMORY
// ============================================================================
dataTable *readCSVToDataTable(FILE *src);
bool writeDataTableToCSV(const dataTable *data, FILE *dst);
void destroyDataTable(dataTable *dt);

// ============================================================================
// PREPROCESS
// ============================================================================
bool dataTable_dropSpoiledRowsInPlace(dataTable *data, float missingThreshold);
bool dataTable_dropColInPlace(dataTable *data, size_t targetCol);
bool dataTable_imputeMissingInPlace(dataTable *data, size_t targetCol, imputeStrategy strat); // TODO
bool dataTable_normalizeNumericInPlace(dataTable *data, size_t targetCol, normalizeStrategy strat); // TODO
bool dataTable_clipOutliersInPlace(dataTable *data, size_t targetCol, double zScoreThreshold); // TODO
bool dataTable_bucketizeInPlace(dataTable *data, size_t targetCol, int numBins); // TODO
bool dataTable_polynomialExpandInPlace(dataTable *data, size_t targetCol, int degree); // TODO
bool dataTable_split2(const dataTable *src, float ratio, dataTable **set1, dataTable **set2); // TODO

// ============================================================================
// ROUTING AND SPLITTING
// ============================================================================
bool dataTable_shuffle(dataTable *data); // TODO
bool dataTable_popColInPlace(dataTable *src, size_t extractCol, dataTable **poppedTarget); // TODO
bool dataTable_popCol(const dataTable *src, size_t extractCol, dataTable **remainingTable, dataTable **poppedTarget); // TODO
#endif // C_SV_H