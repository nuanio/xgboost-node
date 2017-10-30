# XGBoost-Node APIs

## Class

<dl>
<dt><a href="#user-content-xgmatrix"><code>XGMatrix</code></a></dt>
<dd></dd>
<dt><a href="#XGModel"><code>XGModel</code></a></dt>
<dd></dd>
<dt><a href="#Result"><code>Result</code></a></dt>
<dd></dd>
</dl>

## Function

<dl>
<dt><a href="#user-content-xgmatrix"><code>XGMatrix</code></a></dt>
<dd></dd>
<dt><a href="#matrix">matrix(data, row, col, missing = NaN)</a> ⇒ <code>XGMatrix</code></dt>
<dd></dd>
<dt><a href="#restoreMatrix">restoreMatrix(file)</a> ⇒ <code>XGMatrix</code></dt>
<dd></dd>
<dt><a href="#matrixFromCSC">matrixFromCSC(data, indptr, indices, n = 0)</a> ⇒ <code>XGMatrix</code></dt>
<dd></dd>
<dt><a href="#matrixFromCSR">matrixFromCSR(data, indptr, indices, n = 0)</a> ⇒ <code>XGMatrix</code></dt>
<dd></dd>
<dt><a href="#XGModel"><code>XGModel</code></a></dt>
<dd></dd>
<dt><a href="#XGModel">XGModel(file)</a> ⇒ <code>XGModel</code></dt>
<dd></dd>
<dt><a href="#user-content-XGModel-predict">XGModel.predict(xgmatrix, mask = 0, ntree = 0)</a> ⇒ <code>Result</code></dt>
<dd></dd>
<dt><a href="#user-content-XGModel-predictAsync">XGModel.predictAsync(xgmatrix, mask = 0, ntree = 0, cb: (err, res: Float32Array | null) => {})</a></dt>
<dd></dd>
<dt><a href="#Result"><code>Result</code></a></dt>
<dd></dd>
</dl>

## `XGMatrix`

**Kind**: object - input matrix for XGModel

| Field | Type | Description |
| --- | --- | --- |
| matrix | `internal` |readonly property|
| error | <code>Error</code> | error status |

<a name="matrix"></a>

## matrix(data, row, col, missing) ⇒ <code>XGMatrix</code>

**Kind**: global function
**Returns**: [<code>XGMatrix</code>](#XGMatrix) - xgboost matrix

| Param | Type | Description |
| --- | --- | --- |
| data | <code>Float32Array</code> | input matrix with row-major order |
| row | <code>Integer</code> | matrix row |
| col | <code>Integer</code> | matrix col |
| missing | <code>Number = NaN</code> | missing value place holder |

```js
const xgboost = require('xgboost');
const input = new Float32Array([
  5.1,  3.5,  1.4,  0.2, // class 0
  6.6,  3. ,  4.4,  1.4, // class 1
  5.9,  3. ,  5.1,  1.8  // class 2
]);
const mat = new xgboost.matrix(input, 3, 4);
```

<a name="XGMatrix.col"></a>

## XGMatrix.col() ⇒ <code>Result</code>
**Kind**: member function
**Returns**: [<code>Result</code>](#Result) - return matrix column size

```js
mat.col();
```

<a name="XGMatrix.row"></a>

## XGMatrix.row() ⇒ <code>Result</code>
**Kind**: member function
**Returns**: [<code>Result</code>](#Result) - return matrix row size

```js
mat.row();
```
<a name="restoreMatrix"></a>

## restoreMatrix(file) ⇒ <code>XGMatrix</code>
**Kind**: global function
**Returns**: [<code>XGMatrix</code>](#XGMatrix) - xgboost matrix

| Param | Type | Description |
| --- | --- | --- |
| file | <code>string</code> | input matrix file path |

```js
const matFromFile = xgboost.restoreMatrix('test/data/xgmatrix.bin');
```

<a name="matrixFromCSC"></a>

## matrixFromCSC(data, indptr, indices, n) ⇒ <code>XGMatrix</code>
**Kind**: global function
**Returns**: [<code>XGMatrix</code>](#XGMatrix) - xgboost matrix

| Param | Type | Description |
| --- | --- | --- |
| data | <code>Float32Array</code> | input matrix |
| indptr | <code>Uint32Array</code> | pointer to col headers |
| indices | <code>Uint32Array</code> | findex |
| n | <code>Integer = 0</code> | number of rows; when it's set to 0, then guess from data |

```js
// [
//   1, 2, 3, 1,
//   0, 1, 2, 3,
//   0, 1, 1, 1,
// ]
const sparseCSC = xgboost.matrixFromCSC(
  new Float32Array([1, 2, 1, 1, 3, 2, 1, 1, 3, 1]),
  new Uint32Array([0, 1, 4, 7, 10]),
  new Uint32Array([0, 0, 1, 2, 0, 1, 2, 0, 1, 2]),
  0);
```
<a name="matrixFromCSR"></a>

## matrixFromCSR(data, indptr, indices, n) ⇒ <code>XGMatrix</code>
**Kind**: global function
**Returns**: [<code>XGMatrix</code>](#XGMatrix) - xgboost matrix

| Param | Type | Description |
| --- | --- | --- |
| data | <code>Float32Array</code> | input matrix |
| indptr | <code>Uint32Array</code> | pointer to row headers |
| indices | <code>Uint32Array</code> | findex |
| n | <code>Integer = 0</code> | number of columns; when it's set to 0, then guess from data |


```js
// [
//   1, 2, 3, 1,
//   0, 1, 2, 3,
//   0, 1, 1, 1,
// ]
const sparseCSR = xgboost.matrixFromCSR(
  new Float32Array([1, 2, 3, 1, 1, 2, 3, 1, 1, 1]),
  new Uint32Array([0, 4, 7, 10]),
  new Uint32Array([0, 1, 2, 3, 1, 2, 3, 1, 2, 3]),
  0);
```

<a name="XGModel"></a>

## `XGModel`

**Kind**: object - Trained XGModel

| Field | Type | Description |
| --- | --- | --- |
| model | `internal` |readonly property|
| error | <code>Error</code> | error status |

## XGModel(file) ⇒ <code>XGModel</code>
**Kind**: global function
**Returns**: [<code>XGModel</code>](#XGModel) - xgboost model

| Param | Type | Description |
| --- | --- | --- |
| file | <code>string</code> | model file path |

```js
const model = xgboost.XGModel('test/data/iris.xg.model');
```

<a name="XGModel.predict" id="XGModel-predict"></a>

## XGModel.predict(xgmatrix, mask = 0, ntree = 0) ⇒ <code>Result</code>

**Kind**: member function
**Returns**: [<code>Result</code>](#Result) - prediction result with `Float32Array`

| Param | Type | Description |
| --- | --- | --- |
| matrix | <code>XGMatrix</code> | input matrix |
| mask | <code>Integer = 0</code> | options taken in prediction, possible values, <br/> 0:normal prediction, <br/>1:output margin instead of transformed value, <br/>2:output leaf index of trees instead of leaf value, note leaf index is unique per tree, <br/>4:output feature contributions to individual predictions |
| ntree | <code>Integer = 0</code> | limit number of trees used for prediction, <br/> this is only valid for boosted trees when the parameter is set to 0, <br/>  we will use all the trees |

```js
model.predict(mat);
```

<a name="XGModel.predictAsync" id="XGModel-predictAsync"></a>

## XGModel.predictAsync(xgmatrix, mask = 0, ntree = 0, cb: (err, res: Float32Array | null) => {})

**Kind**: member function

| Param | Type | Description |
| --- | --- | --- |
| matrix | <code>XGMatrix</code> | input matrix |
| mask | <code>Integer = 0</code> | options taken in prediction, possible values, <br/> 0:normal prediction, <br/>1:output margin instead of transformed value, <br/>2:output leaf index of trees instead of leaf value, note leaf index is unique per tree, <br/>4:output feature contributions to individual predictions |
| ntree | <code>Integer = 0</code> | limit number of trees used for prediction, <br/> this is only valid for boosted trees when the parameter is set to 0, <br/>  we will use all the trees |
| cb    | <code>Function</code> | callback function to accept error status and a Float32Array result |

```js
model.predictAsync(mat, 0, 0, (err, res) => {
  console.log(err);
  console.log(res);
});
```

<a name="Result"></a>

## `Result`

**Kind**: object

| Field | Type | Description |
| --- | --- | --- |
| value | <code>Float32Array \| number</code> | prediction result or method result |
| error | <code>Error</code> | error status |

<a name="XGMatrix"></a>
