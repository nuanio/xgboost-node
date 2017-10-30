## XGBoost-Node

[![Build Status](https://travis-ci.org/nuanio/xgboost-node.svg?branch=master)](https://travis-ci.org/nuanio/xgboost-node) [![NPM version](https://img.shields.io/npm/v/xgboost.svg)](https://www.npmjs.com/package/xgboost) [![codecov](https://codecov.io/gh/nuanio/xgboost-node/branch/master/graph/badge.svg)](https://codecov.io/gh/nuanio/xgboost-node) [![license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)

eXtreme Gradient Boosting Package in Node.js

XGBoost-Node is a Node.js interface of [XGBoost](https://github.com/dmlc/xgboost). XGBoost is a library from [DMLC](http://dmlc.ml/). It is designed and optimized for boosted trees. The underlying algorithm of XGBoost is an extension of the classic gbm algorithm. With multi-threads and regularization, XGBoost is able to utilize more computational power and get a more accurate prediction.

The package is made to run existing XGBoost model with Node.js easily.

### Features

+ Runs XGBoost Model and make predictions in Node.js.

+ Both dense and sparse matrix input are supported, and missing value is handled.

+ Supports Linux, macOS.

### Install

Install from npm

```bash
npm install xgboost
```

Install from GitHub

```bash
git clone --recursive git@github.com:nuanio/xgboost-node.git
npm install
```

### Documentation

+ [Introduction to XGBoost-Node](./doc/intro.md)

+ [APIs Documentation](./doc/api.md)

+ [Unit Tests Cases](./test/base.js)

+ [中文简介](./doc/intro_zh.md)

### Roadmap

+ [x] Matrix API
+ [x] Model API
+ [x] Prediction API
+ [x] Async API
+ [ ] Windows Support
+ [ ] Training API
+ [ ] Visualization API

### Examples

Train a XGBoost model and save to a file, more in [doc](./doc/intro.md#user-content-usage).

Load the model with XGBoost-Node:

```javascript
const xgboost = require('xgboost');
const model = xgboost.XGModel('iris.xg.model');

const input = new Float32Array([
  5.1,  3.5,  1.4,  0.2, // class 0
  6.6,  3. ,  4.4,  1.4, // class 1
  5.9,  3. ,  5.1,  1.8  // class 2
]);

const mat = new xgboost.matrix(input, 3, 4);
console.log(model.predict(mat));
// {
//   value: [
//     0.991, 0.005, 0.004, // class 0
//     0.004, 0.990, 0.006, // class 1
//     0.005, 0.035, 0.960, // class 2
//   ],
//   error: undefined,      // no error
// }

const errModel = xgboost.XGModel('data/empty');
console.log(errModel);
console.log(errModel.predict());
```

# Contributing

Your help and contribution is very valuable. Welcome to submit issue and pull requests. [Learn more](./.github/CONTRIBUTING.md)
