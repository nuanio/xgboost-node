import * as bindings from 'bindings';
import * as fs from 'fs';
import * as async from 'async';

const xgb = bindings('xgboost');

function validateInput(input: any, typeInvalid: boolean, typeError: Error) {
  let value;
  let error;
  if (input instanceof Error) {
    // input is an error, pass though error
    value = undefined;
    error = input;
  } else if (input === undefined || typeInvalid === true) {
    // invalid input type, return type error
    value = undefined;
    error = typeError;
  } else {
    value = input;
    error = undefined;
  }
  return {
    value,
    error
  }
}

// Result Object
/**
 * @property value {Float32Array | number | undefined} - prediction result or method result
 * @property error {Error} - error status
 */
class Result {
  public value: Float32Array | number | undefined;
  public error: Error | undefined;
  public constructor(input: Float32Array | number | Error) {
    const result = validateInput(input,
      !(input instanceof Float32Array) && typeof input !== 'number',
      TypeError('result should be Float32Array or number'));
    this.value = result.value;
    this.error = result.error;
  }
}

export interface Matrix {
  col(): number;
  row(): number;
}

// XGMatrix Object
/**
 * @property matrix {internal} - readonly property
 * @property error {Error} - error status
 */
class XGMatrix {
  readonly matrix: Matrix | undefined;
  public error: Error | undefined;

  public constructor(input: Matrix | Error) {
    const result = validateInput(input,
      !(input instanceof xgb.CXGMatrix),
      TypeError('input should be xgb.CXGMatrix'));
    this.matrix = result.value;
    this.error = result.error;
  }
  // Matrix column size
  /**
   * @return {Result} column size
   */
  public col() {
    if (this.error) {
      return new Result(this.error);
    }
    try {
      return new Result(this.matrix.col());
    } catch (err) {
      return new Result(err);
    }
  }
  // Matrix row size
  /**
   * @return {Result} row size
   */
  public row() {
    if (this.error) {
      return new Result(this.error);
    }
    try {
      return new Result(this.matrix.row());
    } catch (err) {
      return new Result(err);
    }
  }
}

function typeCheckList(inputList: Array<[boolean, Error]>): Error | undefined {
  for (const value of inputList) {
    if (value[0]) return value[1];
  }
  return undefined;
}

// Create xgboost matrix from array
/**
 * @param data {Float32Array} - input matrix with row-major order
 * @param row {Integer} - matrix row
 * @param col {Integer} - matrix col
 * @param missing {Number}[NaN] - missing value place holder
 * @return {XGMatrix} xgboost matrix
 */
export function matrix(data: Float32Array, row: number,
  col: number, missing = NaN): object {
  const typeError =
    typeCheckList([
      [
        !(data instanceof Float32Array),
        TypeError('data should be Float32Array')
      ],
      [
        !(typeof missing === 'number'),
        TypeError('missing should be Number')
      ],
      [
        !Number.isInteger(col) || !Number.isInteger(row),
        TypeError('col and row should be Integer')
      ],
      [
        col * row !== data.length,
        TypeError(`col*row ${col * row} != input length ${data.length}`)
      ]
    ]);
  try {
    return new XGMatrix(typeError
      || new xgb.CXGMatrix(0, data, row, col, missing));
  } catch (err) {
    return new XGMatrix(err);
  }
}

// Create xgboost matrix from file
/**
 * @param file {string} - input matrix file path
 * @return {XGMatrix} xgboost matrix
 */
export function restoreMatrix(file: string): object {
  const typeError =
    typeCheckList([
      [
        typeof file !== 'string',
        TypeError(`${file} is ${typeof file}, but it should be a string`)
      ],
      [
        !fs.existsSync(file),
        TypeError(`${file} does not exist`)
      ],
    ]);
  try {
    return new XGMatrix(typeError || new xgb.CXGMatrix(1, file));
  } catch (err) {
    return new XGMatrix(err);
  }
}

function checkCSCR(data: Float32Array, indptr: Uint32Array,
  indices: Uint32Array, n: number) {
  return typeCheckList([
    [
      !(data instanceof Float32Array),
      TypeError('data should be Float32Array'),
    ],
    [
      !(indptr instanceof Uint32Array),
      TypeError('indptr should be Uint32Array'),
    ],
    [
      !(indices instanceof Uint32Array),
      TypeError('indices should be Uint32Array'),
    ],
    [
      !Number.isInteger(n),
      TypeError('n should be Integer'),
    ],
  ]);
}

// Create xgboost matrix from CSC format
/**
 * @param data {Float32Array} - input matrix
 * @param indptr {Uint32Array} -  pointer to col headers
 * @param indices {Uint32Array} - findex
 * @param n {Integer}[0] - number of rows; when it's set to 0, then guess from data
 * @return {XGMatrix} xgboost matrix
 */
export function matrixFromCSC(data: Float32Array, indptr: Uint32Array,
  indices: Uint32Array, n = 0): object {
  const typeError = checkCSCR(data, indptr, indices, n);
  try {
    return new XGMatrix(typeError || new xgb.CXGMatrix(2, data, indptr, indices, indptr.length, data.length, n));
  } catch (err) {
    return new XGMatrix(err);
  }
}

// Create xgboost matrix from CSC format
/**
 * @param data {Float32Array} - input matrix
 * @param indptr {Uint32Array} -  pointer to row headers
 * @param indices {Uint32Array} - findex
 * @param n {Integer}[0] - number of columns; when it's set to 0, then guess from data
 * @return {XGMatrix} xgboost matrix
 */
export function matrixFromCSR(data: Float32Array, indptr: Uint32Array,
  indices: Uint32Array, n = 0): object {
  const typeError = checkCSCR(data, indptr, indices, n);
  try {
    return new XGMatrix(typeError ||
      new xgb.CXGMatrix(3, data, indptr, indices, indptr.length, data.length, n));
  } catch (err) {
    return new XGMatrix(err);
  }
}

export interface Predictor {
  predict(matrix: object, mask: number, ntree: number): any;
  predictAsync(matrix: object, mask: number, ntree: number,
    callback: (err: Error | null | undefined, res: Float32Array) => any): any;
}

// queue for xgboost task, libuv thread is not safe for xgboost
const taskQueue = function setupQueue() {
  return async.queue((task, callback) => {
    task.model.predictAsync(task.mat, task.mask, task.ntree, (err, res) => {
      task.callback(err, res);
      task = undefined;
      callback();
    });
  }, 1);
}();

// XGModel Object
/**
 * @property model {internal} - private property
 * @property error {Error} - error status
 */
class XGModelBase {
  private model: Predictor | undefined;
  public error: Error | undefined;

  public constructor(model: Predictor | Error) {
    const result = validateInput(model,
      !(model instanceof xgb.CXGModel),
      TypeError('model should be xgb.CXGModel'));
    this.model = result.value;
    this.error = result.error;
  }

  // Predict result
  /**
   * @param xgmatrix {XGMatrix} - input data, created from dataFrom*() functions
   * @param mask {Integer}[0] - options taken in prediction, possible values
   *          0:normal prediction
   *          1:output margin instead of transformed value
   *          2:output leaf index of trees instead of leaf value, note leaf index is unique per tree
   *          4:output feature contributions to individual predictions
   * @param ntree {Integer}[0] - limit number of trees used for prediction, this is only valid for boosted trees
   * when the parameter is set to 0, we will use all the trees
   * @return {Result} predicted values
   */
  public predict(xgmatrix: XGMatrix, mask = 0, ntree = 0): Result {
    const typeError =
      typeCheckList([
        [
          this.error !== undefined,
          this.error,
        ],
        [
          !(xgmatrix instanceof XGMatrix),
          TypeError('input should be XGMatrix'),
        ],
        [
          xgmatrix instanceof XGMatrix && xgmatrix.error !== undefined,
          xgmatrix instanceof XGMatrix && xgmatrix.error,
        ],
        [
          !Number.isInteger(mask)
          || ![0, 1, 2, 4].includes(mask)
          || !Number.isInteger(ntree),
          TypeError('mask and ntree should be Integer'),
        ],
      ]);
    try {
      return new Result(typeError
        || this.model.predict(xgmatrix.matrix, mask, ntree));
    } catch (err) {
      return new Result(err);
    }
  }

  // Predict result - Async
  /**
   * @param xgmatrix {XGMatrix} - input data, created from dataFrom*() functions
   * @param mask {Integer}[0] - options taken in prediction, possible values
   *          0:normal prediction
   *          1:output margin instead of transformed value
   *          2:output leaf index of trees instead of leaf value, note leaf index is unique per tree
   *          4:output feature contributions to individual predictions
   * @param ntree {Integer}[0] - limit number of trees used for prediction, this is only valid for boosted trees
   * when the parameter is set to 0, we will use all the trees
   * @param callback {Function} - callback function
   */
  public predictAsync(xgmatrix: XGMatrix,
    mask = 0,
    ntree = 0,
    callback: (err: Error | null | undefined, res: Float32Array) => any) {
    const typeError =
      typeCheckList([
        [
          this.error !== undefined,
          this.error,
        ],
        [
          !(xgmatrix instanceof XGMatrix),
          TypeError('input should be XGMatrix'),
        ],
        [
          xgmatrix instanceof XGMatrix && xgmatrix.error !== undefined,
          xgmatrix instanceof XGMatrix && xgmatrix.error,
        ],
        [
          !Number.isInteger(mask)
          || ![0, 1, 2, 4].includes(mask)
          || !Number.isInteger(ntree),
          TypeError('mask and ntree should be Integer'),
        ],
        [
          typeof callback !== 'function',
          TypeError('callback should be a Function'),
        ],
      ]);
    if (typeError) {
      return callback(typeError, null);
    }
    taskQueue.push({
      model: this.model,
      mat: xgmatrix.matrix,
      mask,
      ntree,
      callback,
    });
  }
}

// Create xgboost model from file
/**
 * @param file {string} - model file path
 * @return {XGModel} xgboost model
 */
export function XGModel(file: string): object {
  const typeError =
    typeCheckList([
      [
        typeof file !== 'string',
        TypeError(`${file} is ${typeof file}, but it should be a string`),
      ],
      [
        !fs.existsSync(file),
        TypeError(`${file} does not exist`),
      ],
    ]);
  try {
    return new XGModelBase(typeError || new xgb.CXGModel(file));
  } catch (err) {
    return new XGModelBase(err);
  }
}
