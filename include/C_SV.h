#ifndef C_SV_H
#define C_SV_H

#include "tenCor.h"
#include <stdint.h>

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
    IMPUTE_ZERO = 4,
    IMPUTE_LINEAR_INTERPOLATE = 5,
    IMPUTE_KNN = 6
} imputeStrategy;

typedef enum {
    SCALE_MINMAX = 1,
    SCALE_STANDARD = 2,
    SCALE_BOX = 3
} normalizeStrategy;

typedef enum {
    CAP_NORMAL = 1, // usually 3 (3-sigma rule)
    CAP_IQR = 2, // usually 1.5
    CAP_PERCENTILE = 3 // 5% is a good start
} capStrategy;

// ============================================================================
// MEMORY
// ============================================================================
dataTable *readCSVToDataTable(FILE *src);
dataTable *createEmptyDataTable(int dataRows, int dataCols);
dataTable *createDataTable(int dataRows, int dataCols);
bool writeDataTableToCSV(const dataTable *dt, FILE *dst);
void destroyDataTable(dataTable *dt);

// ============================================================================
// PREPROCESS
// ============================================================================
bool dataTable_dropSpoiledRowsInPlace(dataTable *dt, float missingThreshold);
bool dataTable_dropColInPlace(dataTable *dt, size_t targetCol);
bool dataTable_imputeMissingInPlace(dataTable *dt, size_t targetCol, imputeStrategy strat);

// after the imputing process, there should be no NAN in the data table

bool dataTable_normalizeNumericInPlace(dataTable *dt, size_t targetCol, normalizeStrategy strat);
bool dataTable_capOutliersInPlace(dataTable *dt, size_t targetCol, double threshold, capStrategy strat);
bool dataTable_uniformBucketizeInPlace(dataTable *dt, size_t targetCol, int numBins);

// warning: the next two functions may introduce NANs into the table
bool dataTable_expandColumnInPlace(dataTable *dt, size_t sourceCol, double (*function)(double), char *newHeader);
bool dataTable_polynomialExpandInPlace(dataTable *dt, size_t targetCol, int degree);

// ============================================================================
// ROUTING AND SPLITTING
// ============================================================================
bool dataTable_split2(const dataTable *src, float ratio, dataTable **set1, dataTable **set2);
bool dataTable_shuffle(dataTable *dt);
bool dataTable_popColInPlace(dataTable *src, size_t extractCol, dataTable **poppedTarget);
bool dataTable_transferColInPlace(dataTable *src, dataTable *dest, size_t extractColIdx , size_t insertColIdx);
bool dataTable_popCol(const dataTable *src, size_t extractCol, dataTable **remainingTable, dataTable **poppedTarget);

// ============================================================================
// HELPERS
// ============================================================================
void dataTable_seed(uint32_t seed);

#endif // C_SV_H