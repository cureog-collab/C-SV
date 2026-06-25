#include "../include/C_SV.h"
// there will be some copy-and-paste code in here, but I think
// sometimes they are better than breaking up sections into helper functions
// in terms of performance (I could very be wrong).

static uint32_t rng_state = 123456789;

static inline uint32_t randUint32();
static inline uint32_t lemireRand(uint32_t range);

// split data into 2 subsets
bool dataTable_split2(const dataTable *src, float ratio, dataTable **set1, dataTable **set2); // TODO

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

bool dataTable_transferColInPlace(dataTable *src, size_t extractColIdx, dataTable *dest, size_t insertColIdx); // TODO

bool dataTable_popCol(const dataTable *src, size_t extractCol, dataTable **remainingTable, dataTable **poppedTarget); // TODO

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