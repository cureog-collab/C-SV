#include "../include/C_SV.h"

dataTable *readCSVToDataTable(FILE *src)
{
    if (src == NULL)
    {
        printf("Error: Cannot read NULL!\n");
        return NULL;
    }

    // loop though file to find cols and rows of data (ignore headers)
    int dataRows = 0; // separate by '\n'
    int dataCols = 0; // separate by ','
    char buffer[MAX_CHARS_PER_LINE];
    int scanWindow = sizeof(buffer);

    // assume the first row is the headers row
    // will consider adding more guardrails and error checks later
    if (fgets(buffer, scanWindow, src))
    {
        // count cols
        dataCols = 1; // assume at least one col exists
        for (int i = 0; buffer[i] != '\0'; ++i)
        {
            dataCols += (buffer[i] == ',');
        }
    }

    // count rows of actual data
    while (fgets(buffer, scanWindow, src))
    {
        // if the line is not null (\n, \r\n, \0), count it as a row
        if (buffer[0] != '\n' && buffer[0] != '\r' && buffer[0] != '\0')
        {
            dataRows++;
        }
    }

    // create a dataTable iff dataCols > 0 & dataRows > 0
    if (dataCols < 1 || dataRows < 1)
    {
        printf("Error: File is empty or invalid format!\n");
        return NULL;
    }

    dataTable *result = malloc(sizeof(dataTable));
    if (result == NULL)
    {
        printf("Error: Failed to malloc for result dataTable!\n");
        return NULL;
    }
    result->headers = calloc(dataCols, sizeof(char *));
    if (result->headers == NULL)
    {
        printf("Error: Failed to calloc for result->headers!\n");
        free(result);
        return NULL;
    }
    result->elements = malloc(dataCols * sizeof(tensor *));
    if (result->elements == NULL)
    {
        printf("Error: Failed to malloc for reasult->elements!\n");
        free(result->headers);
        free(result);
        return NULL;
    }

    // create column vectors
    int colVectorShape[1] = {dataRows};
    for (int col = 0; col < dataCols; ++col)
    {
        result->elements[col] = createTensor(1, colVectorShape);
        if (result->elements[col] == NULL)
        {
            printf("Error: Failed to create a column vector at column index %d!\n", col);
            for (int rollbackCol = col - 1; rollbackCol >= 0; --rollbackCol)
            {
                destroyTensor(result->elements[rollbackCol]);
            }
            free(result->elements);
            free(result->headers);
            free(result);
            return NULL;
        }
    }
    result->rows = dataRows;
    result->cols = dataCols;

    // scan the whole file once again
    rewind(src);

    // read headers
    if (fgets(buffer, scanWindow, src) == NULL)
    {
        printf("Error: Failed to read headers during Pass 2.\n");
        destroyDataTable(result);
        return NULL; 
    }
    
    // handle \n, \r\n by turning them into \0
    buffer[strcspn(buffer, "\r\n")] = 0;

    // "hash" the headers
    char *header = strtok(buffer, ",");
    int headerCount = 0;
    while (header != NULL && headerCount < dataCols)
    {
        result->headers[headerCount] = malloc((strlen(header) + 1) * sizeof(char));
        if (result->headers[headerCount] == NULL)
        {
            printf("Error: Failed to allocate memory for header %i\n", headerCount);
            for (int rollbackHeader = headerCount - 1; rollbackHeader >= 0; --rollbackHeader)
            {
                free(result->headers[rollbackHeader]);
            }
            free(result->headers);
            for (int col = 0; col < dataCols; ++col)
            {
                destroyTensor(result->elements[col]);
            }
            free(result->elements);
            free(result);
            return NULL;
        }
        strcpy(result->headers[headerCount], header);
        headerCount++;
        header = strtok(NULL, ",");
    }

    // fill in datas for col vectors in elements
    int row = 0;
    while(fgets(buffer, scanWindow, src) && row < dataRows)
    {
        // skip meaningless lines
        if (buffer[0] == '\n' || buffer[0] == '\r' || buffer[0] == '\0')
        {
            continue;
        }

        char *ptr = buffer; // reading pointer
        char *endPtr = NULL; // end pointer

        for (int col = 0; col < dataCols; ++col)
        {
            double value = strtod(ptr, &endPtr);
            result->elements[col]->data[row] = value;
            ptr = endPtr + (*endPtr == ',');
        }
        row++;
    }

    return result;
}

void destroyDataTable(dataTable *dt)
{
    if (dt == NULL) return;
    if (dt->headers != NULL)
    {
        for (int header = 0; header < dt->cols; ++header)
        {
            free(dt->headers[header]);
        }
        free(dt->headers);
    }
    if (dt->elements != NULL)
    {
        for (int col = 0; col < dt->cols; ++col)
        {
            destroyTensor(dt->elements[col]);
        }
        free(dt->elements);
    }
    free(dt);
}