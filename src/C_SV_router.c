#include "../include/C_SV.h"

// split data into 2 subsets
bool dataTable_split2(const dataTable *src, float ratio, dataTable **set1, dataTable **set2); // TODO
bool dataTable_shuffle(dataTable *dt); // TODO
bool dataTable_popColInPlace(dataTable *src, size_t extractCol, dataTable **poppedTarget); // TODO
bool dataTable_popCol(const dataTable *src, size_t extractCol, dataTable **remainingTable, dataTable **poppedTarget); // TODO
