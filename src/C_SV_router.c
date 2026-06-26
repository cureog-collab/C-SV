#include "../include/C_SV.h"
// there will be some copy-and-paste code in here, but I think
// sometimes they are better than breaking up sections into helper functions
// in terms of performance (I could very be wrong).

static uint32_t rng_state = 123456789;

static inline uint32_t randUint32();
static inline uint32_t lemireRand(uint32_t range);

// split data into 2 subsets
bool dataTable_split2(const dataTable *src, float ratio, dataTable **set1, dataTable **set2)
{
    if (src == NULL)
    {
        printf("Error: Cannot split NULL!\n");
        return false;
    }
    if (ratio <= 0 || ratio >= 1)
    {
        printf("Error: Ratio must be a float between 0 and 1.0!\n");
        return false;
    }
    if (set1 == NULL || set2 == NULL)
    {
        printf("Error: Cannot split into NULL!\n");
        return false;
    }
    int srcDataRows = src->rows;
    int srcDataCols = src->cols;

    int set1DataRows = (int)(srcDataRows * ratio);
    int set2DataRows = srcDataRows - set1DataRows;

    *set1 = createEmptyDataTable(set1DataRows,srcDataCols);
    if (*set1 == NULL)
    {
        printf("Error: Failed to create set1 dataTable!\n");
        return false;
    }
    *set2 = createEmptyDataTable(set2DataRows, srcDataCols);
    if (*set2 == NULL)
    {
        printf("Error: Failed to create set2 dataTable!\n");
        destroyDataTable(*set1);
        return false;
    }

    int shape1[1] = {set1DataRows};
    int shape2[1] = {set2DataRows};

    for (int col = 0; col < srcDataCols; ++col)
    {
        // fill headers
        (*set1)->headers[col] = strdup(src->headers[col]);
        if ((*set1)->headers[col] == NULL)
        {
            printf("Error: Failed to duplicate for headers at col %i of set1!\n", col);
            destroyDataTable(*set1);
            destroyDataTable(*set2);
            return false;
        }
        (*set2)->headers[col] = strdup(src->headers[col]);
        if ((*set2)->headers[col] == NULL)
        {
            printf("Error: Failed to duplicate for headers at col %i of set2!\n", col);
            destroyDataTable(*set1);
            destroyDataTable(*set2);
            return false;
        }

        // fill elements
        (*set1)->elements[col] = createTensor(1, shape1);
        if ((*set1)->elements[col] == NULL)
        {
            printf("Error: Failed to create tensor for set1 elements at col %i!\n", col);
            destroyDataTable(*set1);
            destroyDataTable(*set2);
            return false;            
        }
        (*set2)->elements[col] = createTensor(1, shape2);
        if ((*set2)->elements[col] == NULL)
        {
            printf("Error: Failed to create tensor for set2 elements at col %i!\n", col);
            destroyDataTable(*set1);
            destroyDataTable(*set2);
            return false;            
        }

        double *srcDataPtr  = src->elements[col]->data;
        double *set1DataPtr = (*set1)->elements[col]->data;
        double *set2DataPtr = (*set2)->elements[col]->data;

        memcpy(set1DataPtr, srcDataPtr, set1DataRows * sizeof(double));
        memcpy(set2DataPtr, srcDataPtr + set1DataRows, set2DataRows * sizeof(double));
    }

    return true;
}

bool dataTable_shuffle(dataTable *dt)
{
    if (dt == NULL)
    {
        printf("Error: Cannot shuffle NULL!\n");
        return false;
    }
    int dataRowsMinus1 = dt->rows - 1;
    int dataCols = dt->cols;

    // Fisher-Yates
    for (int i = dataRowsMinus1; i > 0; --i)
    {
        // seed to swap rows
        int j = (int)lemireRand((uint32_t)(i + 1));

        if (i == j) continue;

        // swap actual content
        for (int col = 0; col < dataCols; ++col)
        {
            double temp = dt->elements[col]->data[i];
            dt->elements[col]->data[i] = dt->elements[col]->data[j];
            dt->elements[col]->data[j] = temp;
        }
    }

    return true;
}

bool dataTable_popColInPlace(dataTable *src, size_t extractCol, dataTable **poppedTarget)
{
    if (src == NULL)
    {
        printf("Error: Source table is NULL!\n");
        return false;
    }
    if (poppedTarget == NULL)
    {
        printf("Error: Target table is NULL!\n");
        return false;
    }
    if ((int)extractCol >= src->cols)
    {
        printf("Error: Invalid extract col!\n");
        return false;
    }
    int srcDataRows = src->rows;
    int srcDataCols = src->cols;

    *poppedTarget = createEmptyDataTable(srcDataRows, 1);
    if (*poppedTarget == NULL)
    {
        printf("Error: Failed to create popperTarget!\n");
        return false;
    }

    (*poppedTarget)->headers[0] = src->headers[extractCol];
    (*poppedTarget)->elements[0] = src->elements[extractCol];

    memmove(&src->headers[extractCol], &src->headers[extractCol + 1], (srcDataCols - 1 - extractCol) * sizeof(char *));
    memmove(&src->elements[extractCol], &src->elements[extractCol + 1], (srcDataCols - 1 - extractCol) * sizeof(tensor *));

    src->cols--;
    src->headers[src->cols] = NULL; 
    src->elements[src->cols] = NULL;

    return true;
}

bool dataTable_transferColInPlace(dataTable *src, dataTable *dest, size_t extractColIdx, size_t insertColIdx)
{
    if (src == NULL || dest == NULL)
    {
        printf("Error: Cannot transfer col with NULL!\n");
        return false;
    }
    if ((int)extractColIdx >= src->cols)
    {
        printf("Error: Invalid extractColIdx!\n");
        return false;
    }
    if ((int)insertColIdx > dest->cols)
    {
        printf("Error: Invalid insertColIdx!\n");
        return false;
    }
    if (src->rows != dest->rows)
    {
        printf("Error: Tables must have the same number of rows to transfer columns!\n");
        return false;
    }

    // expand memory of dest
    char **newDestHeaders = realloc(dest->headers, (dest->cols + 1) * sizeof(char *));
    if (newDestHeaders == NULL)
    {
        printf("Error: Failed to realloc headers for destination table!\n");
        return false;
    }

    tensor **newDestElements = realloc(dest->elements, (dest->cols + 1) * sizeof(tensor *));
    if (newDestElements == NULL)
    {
        printf("Error: Failed to realloc elements for destination table!\n");
        free(newDestHeaders);
        return false;
    }
    dest->headers = newDestHeaders;
    dest->elements = newDestElements;

    // insertCol
    size_t destShiftCount = dest->cols - insertColIdx;
    if (destShiftCount > 0)
    {
        memmove(&dest->headers[insertColIdx + 1], &dest->headers[insertColIdx], destShiftCount * sizeof(char *));
        memmove(&dest->elements[insertColIdx + 1], &dest->elements[insertColIdx], destShiftCount * sizeof(tensor *));
    }

    dest->headers[insertColIdx] = src->headers[extractColIdx];
    dest->elements[insertColIdx] = src->elements[extractColIdx];
    dest->cols++;

    size_t srcShiftCount = src->cols - 1 - extractColIdx;
    if (srcShiftCount > 0)
    {
        memmove(&src->headers[extractColIdx], &src->headers[extractColIdx + 1], srcShiftCount * sizeof(char *));
        memmove(&src->elements[extractColIdx], &src->elements[extractColIdx + 1], srcShiftCount * sizeof(tensor *));
    }

    src->cols--;
    src->headers[src->cols] = NULL;
    src->elements[src->cols] = NULL;

    return true;
}

bool dataTable_popCol(const dataTable *src, size_t extractCol, dataTable **remainingTable, dataTable **poppedTarget)
{
    if (src == NULL || remainingTable == NULL || poppedTarget == NULL)
    {
        printf("Error: Cannot perform popCol on NULL!\n");
        return false;
    }
    if ((int)extractCol >= src->cols)
    {
        printf("Error: Invalid extractCol!\n");
        return false;
    }
    int srcDataRows = src->rows;
    int srcDataCols = src->cols;
    int remainDataCols = srcDataCols - 1;

    *poppedTarget = createEmptyDataTable(srcDataRows, 1);
    if (*poppedTarget == NULL)
    {
        printf("Error: Failed to create poppedTarget!\n");
        return false;
    }

    if (remainDataCols > 0)
    {
        *remainingTable = createEmptyDataTable(srcDataRows, remainDataCols);
        if (*remainingTable == NULL)
        {
            destroyDataTable(*poppedTarget);
            return false;
        }
    }
    else
    {
        *remainingTable = NULL; 
    }

    int shape[1] = {srcDataRows};

    (*poppedTarget)->headers[0] = strdup(src->headers[extractCol]);
    (*poppedTarget)->elements[0] = createTensor(1, shape);
    if ((*poppedTarget)->headers[0] == NULL || (*poppedTarget)->elements[0] == NULL)
    {
        destroyDataTable(*poppedTarget);
        if (*remainingTable != NULL) destroyDataTable(*remainingTable);
        return false;
    }
    memcpy((*poppedTarget)->elements[0]->data, src->elements[extractCol]->data, srcDataRows * sizeof(double));

    if (remainDataCols > 0)
    {
        int destCol = 0;
        for (int srcCol = 0; srcCol < srcDataCols; ++srcCol)
        {
            if (srcCol == (int)extractCol) continue;

            (*remainingTable)->headers[destCol] = strdup(src->headers[srcCol]);
            (*remainingTable)->elements[destCol] = createTensor(1, shape);

            if ((*remainingTable)->headers[destCol] == NULL || (*remainingTable)->elements[destCol] == NULL)
            {
                destroyDataTable(*remainingTable);
                destroyDataTable(*poppedTarget);
                return false;
            }

            memcpy((*remainingTable)->elements[destCol]->data, src->elements[srcCol]->data, srcDataRows * sizeof(double));
            destCol++;
        }
    }

    return true;
}

// HERLPERS

// the below algorithm is from DANIEL LEMIRE
// "Fast Random Integer Generation in an Interval"
// ============================================================================
void dataTable_seed(uint32_t seed)
{
    rng_state = (seed != 0) ? seed : 123456789;
}

// source of uniformly-distributed random integers in [0, 2^32)
static inline uint32_t randUint32()
{
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

static inline uint32_t lemireRand(uint32_t range)
{
    // L = 32

    // x <- random integer in [0, 2^L)
    uint32_t x = randUint32();
    
    // m <- x * s
    uint64_t m = (uint64_t)x * (uint64_t)range;
    
    // l <- m mod 2^L
    uint32_t l = (uint32_t)m;
    
    // if l < s then
    if (l < range) 
    {
        // t <- (2^L - s) mod s
        uint32_t t = -range % range;
        
        // while l < t do
        while (l < t) 
        {
            // x <- random integer
            x = randUint32();
            
            // m <- x * s
            m = (uint64_t)x * (uint64_t)range;
            
            // l <- m mod 2^L
            l = (uint32_t)m;
        }
    }
    
    // return m div 2^L
    return (uint32_t)(m >> 32);
}
// ============================================================================