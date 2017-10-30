const xgb = require('../index');
const expect = require('chai').expect;
const async = require('async');

describe('base', () => {
  it('loads a model', () => {
    const m = xgb.XGModel('test/data/iris.xg.model');
    expect(m.model.constructor.name).to.be.equal('CXGModel');
  });

  it('model file not existed', () => {
    const m = xgb.XGModel('data/empty.model');
    expect(m.error).to.be.an('error');
  });

  it('build matrix from typed array', () => {
    // rows = 1, cols = 56, missing = NaN
    const input = new Float32Array(1 * 56);
    input.fill(0.5);
    const mat = xgb.matrix(input, 1, 56, NaN);
    expect(mat.constructor.name, 'class name').to.be.equal('XGMatrix');
    expect(mat.col().value, 'matrix col').to.be.equal(56);
    expect(mat.row().value, 'matrix row').to.be.equal(1);
  });

  it('matrix row col error', () => {
    const input = new Float32Array(2 * 3);
    input.fill(0.5);
    const mat = xgb.matrix(input, 1, 2, NaN);
    expect(mat.error).to.be.an('error');
    expect(mat.col().error).to.be.an('error');
    expect(mat.row().error).to.be.an('error');
  });

  it('loads a matrix from file', () => {
    const mat = xgb.restoreMatrix('test/data/xgmatrix.bin');
    expect(mat.constructor.name, 'class name').to.be.equal('XGMatrix');
    expect(mat.col().value, `matrix col ${mat.col().value}`).to.be.equal(4);
    expect(mat.row().value, `matrix row ${mat.row().value}`).to.be.equal(150);
  });

  it('multi rows matrix', () => {
    // rows = 3, cols = 56, missing is default = NaN
    const input = new Float32Array(3 * 56);
    input.fill(0.5);
    input.fill(0.3, 56, 112);
    input.fill(0.8, 112, 168);
    const mat = new xgb.matrix(input, 3, 56);
    expect(mat.constructor.name, 'class name').to.be.equal('XGMatrix');
    expect(mat.col().value, 'matrix col').to.be.equal(56);
    expect(mat.row().value, 'matrix row').to.be.equal(3);
  });

  it('prediction', () => {
    // iris model, checkout readme for the code to build model
    const m = xgb.XGModel('test/data/iris.xg.model');
    // three samples, one sample has 4 cols.
    const input = new Float32Array([
      5.1, 3.5, 1.4, 0.2, // class 0
      6.6, 3.0, 4.4, 1.4, // class 1
      5.9, 3.0, 5.1, 1.8, // class 2
    ]);
    const mat = new xgb.matrix(input, 3, 4, NaN);
    const result = m.predict(mat);
    // [
    //   0, 1, 2,
    //   3, 4, 5,
    //   6, 7, 8,
    // ]
    expect(result.value[0] > 0.9).to.be.true; // class 0
    expect(result.value[4] > 0.9).to.be.true; // class 1
    expect(result.value[8] > 0.9).to.be.true; // class 2
  });

  it('sparse matrix', () => {
    const dense = new Float32Array([
      1, 2, 3, 1,
      0, 1, 2, 3,
      0, 1, 1, 1,
    ]);

    const sparseCSR = xgb.matrixFromCSR(
      new Float32Array([1, 2, 3, 1, 1, 2, 3, 1, 1, 1]),
      new Uint32Array([0, 4, 7, 10]),
      new Uint32Array([0, 1, 2, 3, 1, 2, 3, 1, 2, 3]),
      0);
    const sparseCSC = xgb.matrixFromCSC(
      new Float32Array([1, 2, 1, 1, 3, 2, 1, 1, 3, 1]),
      new Uint32Array([0, 1, 4, 7, 10]),
      new Uint32Array([0, 0, 1, 2, 0, 1, 2, 0, 1, 2]),
      0);
    const denseMatrix = xgb.matrix(dense, 3, 4, NaN);
    const m = xgb.XGModel('test/data/iris.xg.model');
    const denseRes = m.predict(denseMatrix).value;
    console.log(denseRes);
    const csrRes = m.predict(sparseCSR).value;
    const cscRes = m.predict(sparseCSC).value;
    denseRes.forEach((v, index) => {
      expect(v, `csc ${index}`).equal(cscRes[index]);
      expect(v, `csr ${index}`).equal(csrRes[index]);
    });
  });

  it('sparse matrix error', () => {
    const sparseCSC = xgb.matrixFromCSC(
      new Float32Array([1, 2, 1, 1, 3, 2, 1, 1, 3, 1]),
      new Uint32Array([0, 1, 4, 7, 10]),
      new Uint32Array([0, 0, 1, 2, 0, 1, 2, 0, 1, 2]),
      2);
    expect(sparseCSC.error).to.be.an('error');
  });

  it('prediction error', () => {
    const m = xgb.XGModel('test/data/iris.xg.model');
    const input = undefined;
    const result = m.predict(input);
    expect(result.error).to.be.an('error');
  });

  it('async predict', (done) => {
    const m = xgb.XGModel('test/data/iris.xg.model');
    const input = new Float32Array([
      5.1, 3.5, 1.4, 0.2, // class 0
      6.6, 3.0, 4.4, 1.4, // class 1
      5.9, 3.0, 5.1, 1.8, // class 2
    ]);
    const mat = new xgb.matrix(input, 3, 4, NaN);
    m.predictAsync(mat, 0, 0, (err, result) => {
      // [
      //   0, 1, 2,
      //   3, 4, 5,
      //   6, 7, 8,
      // ]
      expect(err).to.be.null;
      expect(result[0] > 0.9).to.be.true; // class 0
      expect(result[4] > 0.9).to.be.true; // class 1
      expect(result[8] > 0.9).to.be.true; // class 2
      done();
    });
  });

  it('async parallel predict', function (done) {
    this.timeout(120000);
    const m = xgb.XGModel('test/data/iris.xg.model');
    const input = [
      { array: [5.1, 3.5, 1.4, 0.2], type: 0 }, // class 0
      { array: [6.6, 3.0, 4.4, 1.4], type: 1 }, // class 1
      { array: [5.9, 3.0, 5.1, 1.8], type: 2 }, // class 2
    ];
    let i = 0;
    tasks = [];
    function checkAsync(mat, mask, ntree, type, callback) {
      mat = new xgb.matrix(mat, 1, 4, NaN);
      m.predictAsync(mat, mask, ntree, (err, res) => {
        expect(err).to.be.null;
        expect(res[type] > 0.9);
        callback();
      });
    }
    while (i < 5000) {
      input.forEach((v) => {
        const task = async.apply(checkAsync,
          new Float32Array(v.array), 0, 0, v.type
        );
        tasks.push(task);
      });
      i++;
    }
    async.parallelLimit(tasks, 1000, (err) => {
      console.log('async.parallel tasks end');
      expect(err).to.be.null;
      done();
    });
    console.log('async.parallel tasks begin');
  });

  it('async error', function (done) {
    const m = xgb.XGModel('test/data/iris.xg.model');
    m.predictAsync(null, null, null, (err, res) => {
      expect(err).to.be.an('Error');
      done();
    });
  });

  // bugs in xgboost
  it.skip('restore matrix error', function () {
    const m = xgb.restoreMatrix('test/data/iris.xg.model');
    expect(m.error).to.be.an('Error');
  });
});
