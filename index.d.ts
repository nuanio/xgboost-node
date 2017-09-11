export interface Matrix {
    col(): number;
    row(): number;
}
/**
 * @param data {Float32Array} - input matrix with row-major order
 * @param row {Integer} - matrix row
 * @param col {Integer} - matrix col
 * @param missing {Number}[NaN] - missing value place holder
 * @return {XGMatrix} xgboost matrix
 */
export declare function matrix(data: Float32Array, row: number, col: number, missing?: number): object;
/**
 * @param file {string} - input matrix file path
 * @return {XGMatrix} xgboost matrix
 */
export declare function restoreMatrix(file: string): object;
/**
 * @param data {Float32Array} - input matrix
 * @param indptr {Uint32Array} -  pointer to col headers
 * @param indices {Uint32Array} - findex
 * @param n {Integer}[0] - number of rows; when it's set to 0, then guess from data
 * @return {XGMatrix} xgboost matrix
 */
export declare function matrixFromCSC(data: Float32Array, indptr: Uint32Array, indices: Uint32Array, n?: number): object;
/**
 * @param data {Float32Array} - input matrix
 * @param indptr {Uint32Array} -  pointer to row headers
 * @param indices {Uint32Array} - findex
 * @param n {Integer}[0] - number of columns; when it's set to 0, then guess from data
 * @return {XGMatrix} xgboost matrix
 */
export declare function matrixFromCSR(data: Float32Array, indptr: Uint32Array, indices: Uint32Array, n?: number): object;
export interface Predictor {
    predict(matrix: object, mask: number, ntree: number): any;
    predictAsync(matrix: object, mask: number, ntree: number, callback: (err: Error | null | undefined, res: Float32Array) => any): any;
}
/**
 * @param file {string} - model file path
 * @return {XGModel} xgboost model
 */
export declare function XGModel(file: string): object;
