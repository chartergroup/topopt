// Minimal ctypes shim around Apple Accelerate sparse Cholesky (SparseFactor).
// Build: xcrun clang -O2 -shared -fPIC -framework Accelerate -o libaccel_chol.dylib accel_chol.c
#include <Accelerate/Accelerate.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    SparseOpaqueFactorization_Double f;
    long *colStarts;
    int *rowIdx;
    double *vals;
    int bs;
} AFHandle;

// order_method: 0 default, 1 user (use `order`), 2 AMD, 3 METIS, 4 COLAMD.
// Lower triangle of a symmetric matrix in CSC form (0-based). Returns NULL on failure.
// With bs > 1, n counts blocks, nnz counts blocks, vals holds bs*bs numbers per block
// (column-major inside the block). af_solve takes the scalar size n (blocks * bs).
void *af_factor_ex(int n, const long *colStarts, const int *rowIdx, const double *vals,
                   int order_method, const int *order, int bs, int *status) {
    long nnz = colStarts[n];
    long nv = nnz * bs * bs;
    AFHandle *h = (AFHandle *)calloc(1, sizeof(AFHandle));
    h->colStarts = (long *)malloc(sizeof(long) * (n + 1));
    h->rowIdx = (int *)malloc(sizeof(int) * nnz);
    h->vals = (double *)malloc(sizeof(double) * nv);
    h->bs = bs;
    memcpy(h->colStarts, colStarts, sizeof(long) * (n + 1));
    memcpy(h->rowIdx, rowIdx, sizeof(int) * nnz);
    memcpy(h->vals, vals, sizeof(double) * nv);
    SparseAttributes_t att;
    memset(&att, 0, sizeof(att));
    att.kind = SparseSymmetric;
    att.triangle = SparseLowerTriangle;
    SparseMatrixStructure s;
    memset(&s, 0, sizeof(s));
    s.rowCount = n;
    s.columnCount = n;
    s.columnStarts = h->colStarts;
    s.rowIndices = h->rowIdx;
    s.attributes = att;
    s.blockSize = bs;
    SparseMatrix_Double A;
    A.structure = s;
    A.data = h->vals;
    SparseSymbolicFactorOptions so = {
        .control = SparseDefaultControl, .orderMethod = (SparseOrder_t)order_method,
        .order = (int *)order, .ignoreRowsAndColumns = NULL, .malloc = malloc, .free = free,
        .reportError = NULL };
    SparseNumericFactorOptions no = {
        .control = SparseDefaultControl, .scalingMethod = SparseScalingDefault,
        .scaling = NULL, .pivotTolerance = 0.01, .zeroTolerance = 1e-4 * __DBL_EPSILON__ };
    h->f = SparseFactor(SparseFactorizationCholesky, A, so, no);
    *status = (int)h->f.status;
    if (h->f.status != SparseStatusOK) {
        SparseCleanup(h->f);
        free(h->colStarts); free(h->rowIdx); free(h->vals); free(h);
        return NULL;
    }
    return h;
}

// In-place solve; B is column-major n x nrhs.
void af_solve(void *hp, int n, int nrhs, double *B) {
    AFHandle *h = (AFHandle *)hp;
    DenseMatrix_Double X;
    memset(&X, 0, sizeof(X));
    X.rowCount = n;
    X.columnCount = nrhs;
    X.columnStride = n;
    X.data = B;
    SparseSolve(h->f, X);
}

void af_free(void *hp) {
    AFHandle *h = (AFHandle *)hp;
    if (!h) return;
    SparseCleanup(h->f);
    free(h->colStarts); free(h->rowIdx); free(h->vals); free(h);
}

// Old entry point: default ordering.
void *af_factor(int n, const long *colStarts, const int *rowIdx, const double *vals, int *status) {
    return af_factor_ex(n, colStarts, rowIdx, vals, 0, NULL, 1, status);
}

// New numbers, same pattern. Reuses the symbolic part. Returns 0 on success.
int af_refactor(void *hp, const double *vals) {
    AFHandle *h = (AFHandle *)hp;
    long nnz = h->colStarts[h->f.symbolicFactorization.columnCount];
    memcpy(h->vals, vals, sizeof(double) * nnz * h->bs * h->bs);
    SparseMatrix_Double A;
    memset(&A, 0, sizeof(A));
    A.structure.rowCount = h->f.symbolicFactorization.rowCount;
    A.structure.columnCount = h->f.symbolicFactorization.columnCount;
    A.structure.columnStarts = h->colStarts;
    A.structure.rowIndices = h->rowIdx;
    A.structure.attributes.kind = SparseSymmetric;
    A.structure.attributes.triangle = SparseLowerTriangle;
    A.structure.blockSize = h->bs;
    A.data = h->vals;
    SparseRefactor(A, &h->f);
    return (int)h->f.status;
}
