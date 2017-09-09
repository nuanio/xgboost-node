# An Introduction to XGBoost-Node

September 1, 2017

## Introduction

XGBoost-Node is the first port of [XGBoost](https://github.com/dmlc/xgboost) to run existing XGBoost model with Node.js.

XGBoost is a library from [DMLC](http://dmlc.ml/). It is designed and optimized for boosted trees. The underlying algorithm of XGBoost is an extension of the classic gbm algorithm. With multi-threads and regularization, XGBoost is able to utilize more computational power and get a more accurate prediction.

XGBoost-Node requires a C++ compiler to build XGBoost library. Multi-threading version requires a C++ compiler with OpenMP.

## Installation

Install the package from npm

```bash
npm install xgboost
```

Install from GitHub

```bash
npm install https://github.com/nuanio/xgboost-node
```

For multi-threading support, C++ compiler requires OpenMP support. XGBoost-Node will first try to build multi-thread support version with fallback to single thread support.

## Usage

Prepare an XGBoost model, and then save the model file for Node.js.

```python
#!/usr/bin/python3
import xgboost as xgb
from sklearn import datasets

iris = datasets.load_iris()
dtrain = xgb.DMatrix(iris.data, label = iris.target)
param = {
    'max_depth': 3,
    'eta': 0.3,
    'objective': 'multi:softprob',
    'num_class': 3 # three class 0, 1, 2
}
num_round = 20
bst = xgb.train(param, dtrain, num_round)
bst.save_model("iris.xg.model")
```

In Node.js, load existing model file with [XGModel(file)](./api.md#XGModel),

```js
const xgboost = require('xgboost');
const model = xgboost.XGModel('iris.xg.model');
if (model.error) {
    console.log(model.error);
} else {
    console.log("loaded model.")
}
```

To run prediction with the model, create an the input matrix with [matrix(data : Float32Array, row : Integer, col : Integer, missing = NaN)](./api.md#matrix).

```js
const input = new Float32Array([
  5.1,  3.5,  1.4,  0.2, // class 0
  6.6,  3. ,  4.4,  1.4, // class 1
  5.9,  3. ,  5.1,  1.8  // class 2
]);

const mat = new xgboost.matrix(input, 3, 4);
console.log(model.predict(mat));
```

`input` is a Float32Array with a row-major order, the consecutive elements of a row reside next to each other.

The default missing value in the matrix is NaN.

If an XGMatrix creation fails, the matErr and the prediction result will contain an error field.

```js
const matErr = new xgboost.matrix(input, 3, 0); // wrong cols
const errRes = model.predict(matErr);
console.log(errRes.value);
console.log(errRes.error.message);
```

Input matrix can also be sparse matrix with [matrixFromCSC](./api.md#matrixFromCSC) or [matrixFromCSR](./api.md#matrixFromCSR)

```js
const dense = new Float32Array([
    1, 2, 3, 1,
    0, 1, 2, 3,
    0, 1, 1, 1,
]);
const denseMat = xgboost.matrix(dense, 3, 4);

const sparseCSC = xgboost.matrixFromCSC(
  new Float32Array([1, 2, 1, 1, 3, 2, 1, 1, 3, 1]),
  new Uint32Array([0, 1, 4, 7, 10]),
  new Uint32Array([0, 0, 1, 2, 0, 1, 2, 0, 1, 2]),
  0);

console.log(sparseCSC.cols());
console.log(sparseCSC.rows());
model.predict(sparseCSC)
model.predict(denseMat)
```

Learn more about [APIs](./api.md)

## Contributing

[nuan.io](https://nuan.io) is using XGBoost-Node to run machine learning model on Node.js platform.

We are planning to add Async API. For more info, please check out the [roadmap](https://github.com/nuanio/xgboost-node#user-content-roadmap).

Your help and contribution is very valuable to make XGBoost-Node better, welcome to submit issues and pull requests. [Learn more](../.github/CONTRIBUTING.md)
