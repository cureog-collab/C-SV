#include "../include/C_SV.h"

// drop rows exceeding a missing value % threshold
bool dataTable_dropSpoiledRowsInPlace(dataTable *data, float missingThreshold)
{
    if (data == NULL || data->elements == NULL || data->cols <= 0)
    {
        printf("Error: Broken input data!\n");
        return false;
    }
    int dataRows = data->rows;
    int dataCols = data->cols;

    int maxAllowedMissingCols = (int)(data->cols * missingThreshold);
    int writeRow = 0;

    // scan rows
    for (int readRow = 0; readRow < dataRows; ++readRow)
    {
        int missingCol = 0;

        // scan cols
        for (int col = 0; col < dataCols; ++col)
        {
            double value = data->elements[col]->data[readRow];
            missingCol += isnan(value);
        }
        
        // safe row
        if (missingCol <= maxAllowedMissingCols)
        {
            for (int col = 0; col < dataCols; ++col)
            {
                data->elements[col]->data[writeRow] = data->elements[col]->data[readRow];
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

    data->rows = writeRow;

    return true;
}

// remove a feature column
bool dataTable_dropColInPlace(dataTable *data, size_t targetCol)
{
    if (data == NULL || data->headers == NULL || data->elements == NULL)
    {
        printf("Error: Invalid format of input data!\n");
        return false;
    }
    else if ((int)targetCol >= data->cols)
    {
        printf("Error: Target col is greater than data->cols!\n");
        return false;
    }
    int dataColsMinus1 = data->cols - 1;

    free(data->headers[targetCol]);
    destroyTensor(data->elements[targetCol]);

    size_t shiftCount = dataColsMinus1 - targetCol;
    if (shiftCount > 0)
    {
        memmove(&data->headers[targetCol], &data->headers[targetCol + 1], shiftCount * sizeof(char *));
        memmove(&data->elements[targetCol], &data->elements[targetCol + 1], shiftCount * sizeof(tensor *));
    }
    data->cols--;

    return true;
}

// fill missing data using one of the strats in enum imputeStrategy
bool dataTable_imputeMissingInPlace(dataTable *data, size_t targetCol, imputeStrategy strat); // TODO

// normalize numeric range
bool dataTable_normalizeNumericInPlace(dataTable *data, size_t targetCol, normalizeStrategy strat); // TODO

// remove extreme outliers user z-scores
bool dataTable_clipOutliersInPlace(dataTable *data, size_t targetCol, double zScoreThreshold); // TODO

// put cotinuous data into buckets
bool dataTable_bucketizeInPlace(dataTable *data, size_t targetCol, int numBins); // TODO

// generate data cols via polynomial expansion
bool dataTable_polynomialExpandInPlace(dataTable *data, size_t targetCol, int degree); // TODO

// split data into 2 subsets
bool dataTable_split2(const dataTable *src, float ratio, dataTable **set1, dataTable **set2); // TODO