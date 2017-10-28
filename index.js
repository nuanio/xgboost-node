"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var bindings = require("bindings");
var fs = require("fs");
var async = require("async");
var xgb = bindings('xgboost');
function validateInput(input, typeInvalid, typeError) {
    var value;
    var error;
    if (input instanceof Error) {
        // input is an error, pass though error
        value = undefined;
        error = input;
    }
    else if (input === undefined || typeInvalid === true) {
        // invalid input type, return type error
        value = undefined;
        error = typeError;
    }
    else {
        value = input;
        error = undefined;
    }
    return {
        value: value,
        error: error
    };
}
// Result Object
/**
 * @property value {Float32Array | number | undefined} - prediction result or method result
 * @property error {Error} - error status
 */
var Result = (function () {
    function Result(input) {
        var result = validateInput(input, !(input instanceof Float32Array) && typeof input !== 'number', TypeError('result should be Float32Array or number'));
        this.value = result.value;
        this.error = result.error;
    }
    return Result;
}());
// XGMatrix Object
/**
 * @property matrix {internal} - readonly property
 * @property error {Error} - error status
 */
var XGMatrix = (function () {
    function XGMatrix(input) {
        var result = validateInput(input, !(input instanceof xgb.CXGMatrix), TypeError('input should be xgb.CXGMatrix'));
        this.matrix = result.value;
        this.error = result.error;
    }
    // Matrix column size
    /**
     * @return {Result} column size
     */
    XGMatrix.prototype.col = function () {
        if (this.error) {
            return new Result(this.error);
        }
        try {
            return new Result(this.matrix.col());
        }
        catch (err) {
            return new Result(err);
        }
    };
    // Matrix row size
    /**
     * @return {Result} row size
     */
    XGMatrix.prototype.row = function () {
        if (this.error) {
            return new Result(this.error);
        }
        try {
            return new Result(this.matrix.row());
        }
        catch (err) {
            return new Result(err);
        }
    };
    return XGMatrix;
}());
function typeCheckList(inputList) {
    for (var _i = 0, inputList_1 = inputList; _i < inputList_1.length; _i++) {
        var value = inputList_1[_i];
        if (value[0])
            return value[1];
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
function matrix(data, row, col, missing) {
    if (missing === void 0) { missing = NaN; }
    var typeError = typeCheckList([
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
            TypeError("col*row " + col * row + " != input length " + data.length)
        ]
    ]);
    try {
        return new XGMatrix(typeError
            || new xgb.CXGMatrix(0, data, row, col, missing));
    }
    catch (err) {
        return new XGMatrix(err);
    }
}
exports.matrix = matrix;
// Create xgboost matrix from file
/**
 * @param file {string} - input matrix file path
 * @return {XGMatrix} xgboost matrix
 */
function restoreMatrix(file) {
    var typeError = typeCheckList([
        [
            typeof file !== 'string',
            TypeError(file + " is " + typeof file + ", but it should be a string")
        ],
        [
            !fs.existsSync(file),
            TypeError(file + " does not exist")
        ],
    ]);
    try {
        return new XGMatrix(typeError || new xgb.CXGMatrix(1, file));
    }
    catch (err) {
        return new XGMatrix(err);
    }
}
exports.restoreMatrix = restoreMatrix;
function checkCSCR(data, indptr, indices, n) {
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
function matrixFromCSC(data, indptr, indices, n) {
    if (n === void 0) { n = 0; }
    var typeError = checkCSCR(data, indptr, indices, n);
    try {
        return new XGMatrix(typeError || new xgb.CXGMatrix(2, data, indptr, indices, indptr.length, data.length, n));
    }
    catch (err) {
        return new XGMatrix(err);
    }
}
exports.matrixFromCSC = matrixFromCSC;
// Create xgboost matrix from CSC format
/**
 * @param data {Float32Array} - input matrix
 * @param indptr {Uint32Array} -  pointer to row headers
 * @param indices {Uint32Array} - findex
 * @param n {Integer}[0] - number of columns; when it's set to 0, then guess from data
 * @return {XGMatrix} xgboost matrix
 */
function matrixFromCSR(data, indptr, indices, n) {
    if (n === void 0) { n = 0; }
    var typeError = checkCSCR(data, indptr, indices, n);
    try {
        return new XGMatrix(typeError ||
            new xgb.CXGMatrix(3, data, indptr, indices, indptr.length, data.length, n));
    }
    catch (err) {
        return new XGMatrix(err);
    }
}
exports.matrixFromCSR = matrixFromCSR;
// queue for xgboost task, libuv thread is not safe for xgboost
var taskQueue = function setupQueue() {
    return async.queue(function (task, callback) {
        task.model.predictAsync(task.mat, task.mask, task.ntree, function (err, res) {
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
var XGModelBase = (function () {
    function XGModelBase(model) {
        var result = validateInput(model, !(model instanceof xgb.CXGModel), TypeError('model should be xgb.CXGModel'));
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
    XGModelBase.prototype.predict = function (xgmatrix, mask, ntree) {
        if (mask === void 0) { mask = 0; }
        if (ntree === void 0) { ntree = 0; }
        var typeError = typeCheckList([
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
        }
        catch (err) {
            return new Result(err);
        }
    };
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
    XGModelBase.prototype.predictAsync = function (xgmatrix, mask, ntree, callback) {
        if (mask === void 0) { mask = 0; }
        if (ntree === void 0) { ntree = 0; }
        var typeError = typeCheckList([
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
            mask: mask,
            ntree: ntree,
            callback: callback,
        });
    };
    return XGModelBase;
}());
// Create xgboost model from file
/**
 * @param file {string} - model file path
 * @return {XGModel} xgboost model
 */
function XGModel(file) {
    var typeError = typeCheckList([
        [
            typeof file !== 'string',
            TypeError(file + " is " + typeof file + ", but it should be a string"),
        ],
        [
            !fs.existsSync(file),
            TypeError(file + " does not exist"),
        ],
    ]);
    try {
        return new XGModelBase(typeError || new xgb.CXGModel(file));
    }
    catch (err) {
        return new XGModelBase(err);
    }
}
exports.XGModel = XGModel;
