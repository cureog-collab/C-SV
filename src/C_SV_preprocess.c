#include "../include/C_SV.h"

// there will be some copy-and-paste codes in here, but I think
// sometimes they are better than breaking up codes into helper functions
// in terms of performance (I could very be wrong).

typedef struct {
    double dist;
    int address;
} neighbor;

static inline int cmpDouble(const void *a, const void *b);
static inline int cmpNeighbors(const void *a, const void *b);
static inline double calculateDist(const dataTable *dt, int rowA, int rowB);

// drop rows exceeding a missing value % threshold
bool dataTable_dropSpoiledRowsInPlace(dataTable *dt, float missingThreshold)
{
    if (dt == NULL || dt->elements == NULL || dt->cols <= 0)
    {
        printf("Error: Broken input data!\n");
        return false;
    }
    int dataRows = dt->rows;
    int dataCols = dt->cols;

    int maxAllowedMissingCols = (int)(dt->cols * missingThreshold);
    int writeRow = 0;

    // scan rows
    for (int readRow = 0; readRow < dataRows; ++readRow)
    {
        int missingCol = 0;

        // scan cols
        for (int col = 0; col < dataCols; ++col)
        {
            double value = dt->elements[col]->data[readRow];
            missingCol += isnan(value);
        }
        
        // safe row
        if (missingCol <= maxAllowedMissingCols)
        {
            for (int col = 0; col < dataCols; ++col)
            {
                dt->elements[col]->data[writeRow] = dt->elements[col]->data[readRow];
            }
            writeRow++;
        }
    }

    // // give back memory, idk if this is needed or not
    // if (writeRow < dataRows)
    // {
    //     for (int col = 0; col < dataCols; ++col)
    //     {
    //         double *shrunkData = realloc(data->elements[col]->data, writeRow * sizeof(double));
    //         if (shrunkData != NULL)
    //         {
    //             data->elements[col]->data = shrunkData;
    //         }
    //     }
    // }

    dt->rows = writeRow;

    return true;
}

// remove a feature column
bool dataTable_dropColInPlace(dataTable *dt, size_t targetCol)
{
    if (dt == NULL || dt->headers == NULL || dt->elements == NULL)
    {
        printf("Error: Invalid format of input data!\n");
        return false;
    }
    else if ((int)targetCol >= dt->cols)
    {
        printf("Error: Target col is greater than dt->cols!\n");
        return false;
    }
    int dataColsMinus1 = dt->cols - 1;

    free(dt->headers[targetCol]);
    destroyTensor(dt->elements[targetCol]);

    size_t shiftCount = dataColsMinus1 - targetCol;
    if (shiftCount > 0)
    {
        memmove(&dt->headers[targetCol], &dt->headers[targetCol + 1], shiftCount * sizeof(char *));
        memmove(&dt->elements[targetCol], &dt->elements[targetCol + 1], shiftCount * sizeof(tensor *));
    }
    dt->cols--;

    return true;
}

// fill missing data using one of the strats in enum imputeStrategy
bool dataTable_imputeMissingInPlace(dataTable *dt, size_t targetCol, imputeStrategy strat)
{
    if (dt == NULL)
    {
        printf("Error: Cannot impute for NULL!\n");
        return false;
    }
    else if ((int)targetCol >= dt->cols)
    {
        printf("Error: Invalid target collumn index!\n");
        return false;
    }
    int dataRows = dt->rows;
    double *dataPtr = dt->elements[targetCol]->data;

    switch (strat)
    {
        // turn NANs to col mean
        case IMPUTE_MEAN:
        {
            double sum = 0.0;
            int validCount = 0;
            for (int row = 0; row < dataRows; ++row)
            {
                if (isnan(dataPtr[row]))
                {
                    continue;
                }
                sum += dataPtr[row];
                validCount++;
            }
            if (validCount == 0)
            {
                printf("Warning: Col %zu is full of NANs! Cannot impute mean.\n", targetCol);
                return false;
            }
            double colMean = sum / validCount;
            for (int row = 0; row < dataRows; ++row)
            {
                if (isnan(dataPtr[row]))
                {
                    dataPtr[row] = colMean;
                }
            }
            break;
        }

        case IMPUTE_MEDIAN:
        {
            double *buffer = malloc(dataRows * sizeof(double));
            if (buffer == NULL)
            {
                printf("Error: Failed to malloc for buffer!\n");
                return false;
            }
            int validCount = 0;
            for (int row = 0; row < dataRows; ++row)
            {
                if (isnan(dataPtr[row])) continue;
                buffer[validCount] = dataPtr[row];
                validCount++;
            }
            if (validCount == 0)
            {
                printf("Warning: Col %zu is full of NANs! Cannot impute median.\n", targetCol);
                free(buffer);
                return false;
            }
            qsort(buffer, validCount, sizeof(double), cmpDouble);
            double colMedian;
            if (validCount & 1)
            {
                colMedian = buffer[validCount >> 1];
            }
            else
            {
                colMedian = (buffer[validCount >> 1] + buffer[(validCount >> 1) - 1]) / 2.0;
            }
            free(buffer);

            for (int row = 0; row < dataRows; ++row)
            {
                if (!isnan(dataPtr[row])) continue;
                dataPtr[row] = colMedian;
            }
            break;
        }

        // turn NANs to col mode
        case IMPUTE_MODE:
        {
            double *buffer = malloc(dataRows * sizeof(double));
            if (buffer == NULL)
            {
                printf("Error: Failed to malloc for buffer!\n");
                return false;
            }
            int validCount = 0;
            for (int row = 0; row < dataRows; ++row)
            {
                if (isnan(dataPtr[row])) continue;
                buffer[validCount] = dataPtr[row];
                validCount++;
            }
            if (validCount == 0)
            {
                printf("Warning: Col %zu is full of NANs! Cannot impute mode.\n", targetCol);
                free(buffer);
                return false;
            }
            qsort(buffer, validCount, sizeof(double), cmpDouble);
            
            int modeIdxInBuffer = 0;
            int currBufferIdx = 0;
            int currBufferIdxCount = 1;
            int maxCount = 1;
            for (int i = 1; i < validCount; ++i)
            {
                if (buffer[i] == buffer[currBufferIdx])
                {
                    currBufferIdxCount++;
                    continue;
                }
                if (currBufferIdxCount > maxCount)
                {
                    maxCount = currBufferIdxCount;
                    modeIdxInBuffer = currBufferIdx;
                }
                currBufferIdxCount = 1;
                currBufferIdx = i;
            }
            if (currBufferIdxCount > maxCount)
            {
                modeIdxInBuffer = currBufferIdx;
            }
            double colMode = buffer[modeIdxInBuffer];
            free(buffer);

            for (int row = 0; row < dataRows; ++row)
            {
                if (!isnan(dataPtr[row])) continue;
                dataPtr[row] = colMode;
            }
            break;
        }

        // turn NANs to 0s
        case IMPUTE_ZERO:
        {
            for (int row = 0; row < dataRows; ++row)
            {
                if (isnan(dataPtr[row])) 
                {
                    dataPtr[row] = 0.0;
                }
            }
            break;
        }

        case IMPUTE_LINEAR_INTERPOLATE:
        {
            int lastValidRow = -1;
            for (int row = 0; row < dataRows; ++row)
            {
                if (!isnan(dataPtr[row]))
                {
                    lastValidRow = row;
                    continue;
                }
                int nextValidRow = -1;

                for (int rowFromNAN = row + 1; rowFromNAN < dataRows; ++rowFromNAN)
                {
                    if (!isnan(dataPtr[rowFromNAN]))
                    {
                        nextValidRow = rowFromNAN;
                        break;
                    }
                }

                // check edge cases
                if (lastValidRow == -1 || nextValidRow == -1)
                {
                    // idk what I should do here
                    continue;
                }

                // linear interpolation
                double y1 = dataPtr[lastValidRow];
                double y2 = dataPtr[nextValidRow];
                int x1 = lastValidRow;
                int x2 = nextValidRow;
                
                double k = (y2 - y1) / (x2 - x1);

                // fill all inbetween NANs
                for (int fillRow = row; fillRow < nextValidRow; ++fillRow)
                {
                    dataPtr[fillRow] = y1 + k * (fillRow - x1);
                }

                row = nextValidRow - 1;
            }
            break;
        }

        case IMPUTE_KNN:
        {
            int K = 5;
            neighbor *neighborhood = malloc(dataRows * sizeof(neighbor));
            if (neighborhood == NULL)
            {
                printf("Error: Failed to malloc for buffer!\n");
                return false;
            }
            bool *isOriginalNAN = malloc(dataRows * sizeof(bool));
            if (isOriginalNAN == NULL)
            {
                printf("Error: Failed to malloc for isOriginalNAN!\n");
                free(isOriginalNAN);
                return false;
            }

            for (int i = 0; i < dataRows; ++i)
            {
                isOriginalNAN[i] = isnan(dataPtr[i]);
            }

            for (int row = 0; row < dataRows; ++row)
            {
                if (!isOriginalNAN[row]) continue;
                int friendlyNeighbors = 0;
                for (int neighborRow = 0; neighborRow < dataRows; ++neighborRow)
                {
                    if (isOriginalNAN[neighborRow]) continue;
                    neighborhood[friendlyNeighbors].dist = calculateDist(dt, row, neighborRow);
                    neighborhood[friendlyNeighbors].address = neighborRow;
                    friendlyNeighbors++;
                }

                // no neighbor is friendly
                if (friendlyNeighbors == 0)
                {
                    printf("Warning: Col %zu is full of NANs! Cannot impute KNN.\n", targetCol);
                    break;
                }

                qsort(neighborhood, friendlyNeighbors, sizeof(neighbor), cmpNeighbors);

                // check if there are enough friendly neighbors
                K = (friendlyNeighbors > K) ? K : friendlyNeighbors;
                
                double sum = 0;
                for (int i = 0; i < K; ++i)
                {
                    int neighborIdxInDt = neighborhood[i].address;
                    sum += dataPtr[neighborIdxInDt];
                }
                dataPtr[row] = sum / K;
            }

            free(neighborhood);
            free(isOriginalNAN);
            break;
        }
    }

    return true;
}

// normalize numeric range
bool dataTable_normalizeNumericInPlace(dataTable *dt, size_t targetCol, normalizeStrategy strat)
{
    if (dt == NULL)
    {
        printf("Error: Cannot normalize NULL!\n");
        return false;
    }
    if ((int)targetCol >= dt->cols)
    {
        printf("Error: Invalid targetCol index!\n");
        return false;
    }
    int dataRows = dt->rows;
    double *dataPtr = dt->elements[targetCol]->data;

    switch (strat)
    {
        case SCALE_MINMAX:
        {
            // find max and min
            double currMax = dataPtr[0];
            double currMin = dataPtr[0];
            for (int row = 1; row < dataRows; ++row)
            {
                if (dataPtr[row] > currMax)
                {
                    currMax = dataPtr[row];
                    continue;
                }
                else if (dataPtr[row] < currMin)
                {
                    currMin = dataPtr[row];
                    continue;
                }
            }
            double max = currMax;
            double min = currMin;

            // check extreme cases
            if (max == min)
            {
                printf("Warning: All data in col %zu is equal! Nothing to scale.\n", targetCol);
                for (int row = 0; row < dataRows; ++row)
                {
                    dataPtr[row] = 0;
                }
                break;
            }

            double invRange = 1.0 / (max - min);

            for (int row = 0; row < dataRows; ++row)
            {
                dataPtr[row] = (dataPtr[row] - min) * invRange;
            }

            break;
        }

        case SCALE_BOX:
        {
            // find Q2, Q1, Q3
            // min -- 25% -- Q1 -- 25% -- Q2 (median) -- 25% -- Q3 -- max

            // sort data
            double *buffer = malloc(dataRows * sizeof(double));
            if (buffer == NULL)
            {
                printf("Error: Failed to malloc for buffer!\n");
                break;
            }
            memcpy(buffer, dataPtr, dataRows * sizeof(double));
            qsort(buffer, dataRows, sizeof(double), cmpDouble);

            double Q1, Q2, Q3;
            int mid = dataRows >> 1;
            if (dataRows & 1)
            {
                Q2 = buffer[mid];
            }
            else
            {
                Q2 = (buffer[mid] + buffer[mid - 1]) / 2.0;
            }

            int upStart = dataRows - mid;
            if (mid & 1)
            {
                Q1 = buffer[mid >> 1];
                Q3 = buffer[upStart + (mid >> 1)];
            }
            else  
            {
                Q1 = (buffer[mid >> 1] + buffer[(mid >> 1) - 1]) / 2.0;
                Q3 = (buffer[upStart + (mid >> 1)] + buffer[upStart - 1 + (mid >> 1)]) / 2.0;
            }

            free(buffer);
            
            if (Q3 == Q1)
            {
                printf("Warning: IQR is 0 in col %zu!\n", targetCol);
                for (int row = 0; row < dataRows; ++row)
                {
                    dataPtr[row] = 0;
                }
                break;
            }

            double invIQR = 1.0 / (Q3 - Q1);
            for (int row = 0; row < dataRows; ++row)
            {
                dataPtr[row] = (dataPtr[row] - Q2) * invIQR;
            }

            break;
        }

        case SCALE_STANDARD:
        {
            double invDataRows = 1.0 / dataRows;
            // find mu
            double sum = dataPtr[0];
            for (int row = 1; row < dataRows; ++row)
            {
                sum += dataPtr[row];
            }
            double mean = sum * invDataRows;

            // find sigma
            double squaredSum = 0;
            for (int row = 0; row < dataRows; ++row)
            {
                squaredSum += (dataPtr[row] - mean) * (dataPtr[row] - mean);
            }

            // check extreme cases
            if (squaredSum == 0)
            {
                printf("Warning: All data in col %zu is equal! Nothing to scale.\n", targetCol);
                for (int row = 0; row < dataRows; ++row)
                {
                    dataPtr[row] = 0;
                }
                break;
            }

            double invDeviation = 1.0 / sqrt(squaredSum * invDataRows);

            // scale
            for (int row = 0; row < dataRows; ++row)
            {
                dataPtr[row] = (dataPtr[row] - mean) * invDeviation;
            }

            break;
        }
    }    

    return true;
}

// remove extreme outliers user z-scores
bool dataTable_capOutliersInPlace(dataTable *dt, size_t targetCol, double threshold, capStrategy strat)
{
    if (dt == NULL)
    {
        printf("Error: Cannot clip outliers in NULL!\n");
        return false;
    }
    if ((int)targetCol > dt->cols)
    {
        printf("Error: Invalid targetCol!\n");
        return false;
    }
    int dataRows = dt->rows;
    int dataCols = dt->cols;
    double *dataPtr = dt->elements[targetCol]->data;

    switch (strat)
    {    
        case CAP_IQR:
        {
            break;
        }

        case CAP_NORMAL:
        {
            break;
        }

        case CAP_PERCENTILE:
        {
            break;
        }
    }

    return true;
}

// put cotinuous data into buckets
bool dataTable_bucketizeInPlace(dataTable *dt, size_t targetCol, int numBins); // TODO

// generate data cols via polynomial expansion
bool dataTable_polynomialExpandInPlace(dataTable *dt, size_t targetCol, int degree); // TODO

// split data into 2 subsets
bool dataTable_split2(const dataTable *src, float ratio, dataTable **set1, dataTable **set2); // TODO

// HELPERS
static inline int cmpDouble(const void *a, const void *b)
{
    const double valA = *(const double *)a;
    const double valB = *(const double *)b;

    // if a = b: 0
    // if a > b: 1
    // if a < b: -1
    if (valA > valB) return 1;
    if (valA < valB) return -1;
    return 0;
}

static inline int cmpNeighbors(const void *a, const void *b)
{
    const double distA = ((const neighbor *)a)->dist;
    const double distB = ((const neighbor *)b)->dist;

    if (distA > distB) return 1;
    if (distA < distB) return -1;
    return 0;
}

static inline double calculateDist(const dataTable *dt, int rowA, int rowB)
{
    double squaredSum = 0;
    int validCount = 0;
    int dataCols = dt->cols;
    for (int col = 0; col < dataCols; ++col)
    {
        double valA = dt->elements[col]->data[rowA];
        double valB = dt->elements[col]->data[rowB];

        if (isnan(valA) || isnan(valB)) continue;

        double diff = valB - valA;
        squaredSum += diff * diff;
        validCount++;
    }

    if (validCount == 0) return INFINITY;

    return sqrt(squaredSum);
}