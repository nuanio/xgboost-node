# XGBoost-Node

2017 年 9 月 1 日

## 简介

XGBoost-Node 是 [XGBoost](https://github.com/dmlc/xgboost) 的第一个 Node.js 库。XGBoost-Node 可以在 Node.js 中运行已经训练好的 XGBoost 模型。

XGBoost 是由 [DMLC](http://dmlc.ml/) 开发的。 XGBoost 算法是传统 gbm 算法的扩展。多线程和正则化使 XGBoost 可以获得更强大的计算性能和更准确的预测结果。


编译 XGBoost 需要 C++ 编译器。编译多线程版本需要 C++ 编译器支持 OpenMP。

## 安装

从 npm 安装

```bash
npm install xgboost
```

从 GitHub 安装

```bash
npm install https://github.com/nuanio/xgboost-node
```

C++ 编译器需要支持 OpenMP 来启用多线程支持。XGBoost-Node 会先尝试构建多线程版本，并在不支持 OpenMP 的系统中构建单线程版本。

## 使用

构建 XGBoost 模型，并将模型文件保存下来供 Node.js 使用。

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

在 Node.js 中, 读取模型文件 [XGModel(file)](./api.md#XGModel),

```js
const xgboost = require('xgboost');
const model = xgboost.XGModel('iris.xg.model');
if (model.error) {
    console.log(model.error);
} else {
    console.log("loaded model.")
}
```

在使用模型进行预测前, 构建输入矩阵 [matrix(data : Float32Array, row : Integer, col : Integer, missing = NaN)](./api.md#matrix).

```js
const input = new Float32Array([
  5.1,  3.5,  1.4,  0.2, // class 0
  6.6,  3. ,  4.4,  1.4, // class 1
  5.9,  3. ,  5.1,  1.8  // class 2
]);

const mat = new xgboost.matrix(input, 3, 4);
console.log(model.predict(mat));
```

`input` 是一个行优先的 Float32Array , 即同一行中元素在数组中相互临近。

矩阵的默认缺失值是 NaN。

如果 XGMatrix 构建失败, 返回的结果 matErr 和预测结果将会包含 error 字段。

```js
const matErr = new xgboost.matrix(input, 3, 0); // wrong cols
const errRes = model.predict(matErr);
console.log(errRes.value);
console.log(errRes.error.message);
```

输入矩阵也可以为稀疏矩阵 [matrixFromCSC](./api.md#matrixFromCSC) 或者 [matrixFromCSR](./api.md#matrixFromCSR).

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

更多 [API 接口文档](./api.md)

## 更多信息

[nuan.io](https://nuan.io) 使用 XGBoost-Node 来运行机器学习模型。

我们计划增加 Async API. 可以查看 [roadmap](https://github.com/nuanio/xgboost-node#user-content-roadmap) 来了解更多的信息。

你的支持能让 XGBoost-Node 更加完善！欢迎提交 issues 和 pull requests。[了解更多](../.github/CONTRIBUTING.md)
