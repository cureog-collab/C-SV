#include "../include/C_SV.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// there will be some copy-and-paste code in here, but I think
// sometimes they are better than breaking up sections into helper functions
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
    if ((int)targetCol >= dt->cols)
    {
        printf("Error: Invalid targetCol!\n");
        return false;
    }
    int dataRows = dt->rows;
    double *dataPtr = dt->elements[targetCol]->data;

    switch (strat)
    {    
        case CAP_IQR:
        {
            double *buffer = malloc(dataRows * sizeof(double));
            if (buffer == NULL)
            {
                printf("Error: Failed to malloc for buffer!\n");
                break;
            }
            memcpy(buffer, dataPtr, dataRows * sizeof(double));
            qsort(buffer, dataRows, sizeof(double), cmpDouble);

            double Q1, Q3;
            int mid = dataRows >> 1;

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

            double IQR = Q3 - Q1;
            double lowerBound = Q1 - threshold * IQR;
            double upperBound = Q3 + threshold * IQR;

            for (int row = 0; row < dataRows; ++row)
            {
                if (dataPtr[row] < lowerBound)
                {
                    dataPtr[row] = lowerBound;
                }
                else if (dataPtr[row] > upperBound)
                {
                    dataPtr[row] = upperBound;
                }
            }
            break;
        }

        case CAP_NORMAL:
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
                printf("Warning: All data in col %zu is equal! No outliers to cap.\n", targetCol);
                break;
            }

            double deviation = sqrt(squaredSum * invDataRows);

            double lowerBound = mean - threshold * deviation;
            double upperBound = mean + threshold * deviation;

            for (int row = 0; row < dataRows; ++row)
            {
                if (dataPtr[row] < lowerBound)
                {
                    dataPtr[row] = lowerBound;
                }
                else if (dataPtr[row] > upperBound)
                {
                    dataPtr[row] = upperBound;
                }
            }

            break;
        }

        case CAP_PERCENTILE:
        {
            if (threshold > 0.5)
            {
                printf("Error: Threshold for winsorization must be smaller than 50%%!\n");
                break;
            }

            double *buffer = malloc(dataRows * sizeof(double));
            if (buffer == NULL)
            {
                printf("Error: Failed to malloc for buffer!\n");
                break;
            }
            memcpy(buffer, dataPtr, dataRows * sizeof(double));
            qsort(buffer, dataRows, sizeof(double), cmpDouble);

            double lowerBound = buffer[(int)(dataRows * threshold)];
            double upperBound = buffer[(int)(dataRows * (1.0 - threshold)) - 1];

            free(buffer);

            for (int row = 0; row < dataRows; ++row)
            {
                if (dataPtr[row] < lowerBound)
                {
                    dataPtr[row] = lowerBound;
                }
                else if (dataPtr[row] > upperBound)
                {
                    dataPtr[row] = upperBound;
                }
            }

            break;
        }
    }

    return true;
}

// put cotinuous data into buckets
bool dataTable_uniformBucketizeInPlace(dataTable *dt, size_t targetCol, int numBins)
{
    if (dt == NULL)
    {
        printf("Error: Cannot bucketize NULL!\n");
        return false;
    }
    if ((int)targetCol >= dt->cols)
    {
        printf("Error: Invalid targetCol!\n");
        return false;
    }
    if (numBins == 0)
    {
        printf("Error: Number of bins must be a finite integer!\n");
        return false;
    }
    int dataRows = dt->rows;
    double *dataPtr = dt->elements[targetCol]->data;

    // find max and min values
    double max = dataPtr[0];
    double min = dataPtr[0];
    for (int row = 0; row < dataRows; ++row)
    {
        if (dataPtr[row] > max)
        {
            max = dataPtr[row];
        }
        else if (dataPtr[row] < min)
        {
            min = dataPtr[row];
        }
    }

    if (max == min)
    {
        printf("Warning: All data in col %zu is equal! Putting all in bin 0.\n", targetCol);
        for (int row = 0; row < dataRows; ++row)
        {
            dataPtr[row] = 0;
        }
        return true;
    }

    // distance between consecutive bins
    double invBinGap = numBins / (max - min);

    for (int row = 0; row < dataRows; ++row)
    {
        double value = dataPtr[row];
        int binIdx = (int)((value - min) * invBinGap);

        if (binIdx >= numBins)
        {
            binIdx = numBins - 1;
        }

        dataPtr[row] = (double)binIdx;
    }

    return true;
}

bool dataTable_expandColumnInPlace(dataTable *dt, size_t sourceCol, double (*function)(double), char *newHeader)
{
    if (dt == NULL)
    {
        printf("Error: Cannot expand NULL!\n");
        return false;
    }
    if ((int)sourceCol >= dt->cols)
    {
        printf("Error: Invalid source col!\n");
        return false;
    }
    if (function == NULL)
    {
        printf("Error: Please input a valid function!\n");
        return false;
    }

    char finalNewHeader[128];
    if (newHeader == NULL)
    {
        printf("Warning: Input newHeader is NULL, setting it to \"col_%i\".", dt->cols);
        snprintf(finalNewHeader, sizeof(finalNewHeader), "col_%i", dt->cols);
    }
    else
    {
        strcpy(finalNewHeader, newHeader);
    }
    
    int dataRows = dt->rows;
    int dataCols = dt->cols;
    double *srcDataPtr = dt->elements[sourceCol]->data;

    // expand the memory of the data table
    char **updatedDataTableHeaders = realloc(dt->headers, (dataCols + 1) * sizeof(char *));
    if (updatedDataTableHeaders == NULL)
    {
        printf("Error: Failed to realloc for updatedDataTableHeaders!\n");
        return false;
    }
    updatedDataTableHeaders[dataCols] = malloc(strlen(finalNewHeader) + 1);
    if (updatedDataTableHeaders[dataCols] == NULL)
    {
        printf("Error: Failed to malloc for new header!\n");
        return false;
    }
    strcpy(updatedDataTableHeaders[dataCols], finalNewHeader);
    dt->headers = updatedDataTableHeaders;

    tensor **newElements = realloc(dt->elements, (dataCols + 1) * sizeof(tensor *));
    if (newElements == NULL)
    {
        printf("Error: Failed to realloc for newElements!\n");
        return false;
    }
    int colVectorShape[1] = {dataRows};
    dt->elements = newElements;
    dt->elements[dataCols] = createTensor(1, colVectorShape);
    dataCols++;
    dt->cols++;

    // apply function to each value in sourceCol
    // and map them to their corresponding address in the new col
    double *newDataPtr = dt->elements[dataCols - 1]->data;
    for (int row = 0; row< dataRows; ++row)
    {
        double newVal = function(srcDataPtr[row]);
        if (isnan(newVal))
        {
            printf("Warning: Element at row %i col %i is NAN!", row, dataCols - 1);
        }
        newDataPtr[row] = newVal;
    }

    return true;
}

// generate data cols via polynomial expansion
bool dataTable_polynomialExpandInPlace(dataTable *dt, size_t targetCol, int degree)
{
    if (dt == NULL)
    {
        printf("Error: Cannot expand NULL!\n");
        return false;
    }
    if ((int)targetCol >= dt->cols)
    {
        printf("Error: Invalid target col!\n");
        return false;
    }
    if (degree <= 1)
    {
        printf("Warning: Input degree is %i, quitting polynomial expansion...", degree);
        return true;
    }
    int dataRows = dt->rows;
    int dataCols = dt->cols;
    int toAddCols = degree - 1;
    double *srcDataPtr = dt->elements[targetCol]->data;

    // new headers and elements
    char **updatedHeaders = realloc(dt->headers, (dataCols + toAddCols) * sizeof(char *));
    if (updatedHeaders == NULL)
    {
        printf("Error: Failed to realloc for new headers!\n");
        return false;
    }
    dt->headers = updatedHeaders;

    tensor **updatedElements = realloc(dt->elements, (dataCols + toAddCols) * sizeof(tensor *));
    if (updatedElements == NULL)
    {
        printf("Error: Failed to realloc for new elements!\n");
        return false;
    }
    dt->elements = updatedElements;

    // generate cols with corresponding degree
    for (int d = 2; d <= degree; ++d)
    {
        int currNewColIdx = dataCols + d - 2;

        // new col name
        char buffer[128];
        snprintf(buffer, sizeof(buffer), "%s^%i", dt->headers[targetCol], d);
        dt->headers[currNewColIdx] = strdup(buffer);

        // new col vector
        int shape[1] = {dataRows};
        dt->elements[currNewColIdx] = createTensor(1, shape);
        if (dt->elements[currNewColIdx] == NULL)
        {
            printf("Error: Failed to create tensor for col %i!\n", currNewColIdx);
            free(dt->headers[currNewColIdx]);
            for (int rollback = currNewColIdx - 1; rollback >= dataCols; --rollback)
            {
                free(dt->headers[rollback]);
                destroyTensor(dt->elements[rollback]);
            }
            dt->headers = realloc(dt->headers, dataCols * sizeof(char *));
            dt->elements = realloc(dt->elements, dataCols * sizeof(tensor *));

            return false;
        }

        double *prevDataPtr = (d == 2) ? srcDataPtr : dt->elements[currNewColIdx - 1]->data;
        double *currDataPtr = dt->elements[currNewColIdx]->data;
        for (int row = 0; row < dataRows; ++row)
        {
            currDataPtr[row] = prevDataPtr[row] * srcDataPtr[row];
        }
    }
    dt->cols += toAddCols;
    
    return true;
}

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