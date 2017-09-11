#ifndef XGMODEL_H
#define XGMODEL_H

#include "base.h"

class XGModel : public Nan::ObjectWrap
{
public:
  static void Init(v8::Local<v8::Object> exports);

private:
  explicit XGModel(BoosterHandle result);
  ~XGModel();

  static NAN_METHOD(New);
  static NAN_METHOD(Predict);
  static NAN_METHOD(PredictAsync);
  static Nan::Persistent<v8::Function> constructor;
  BoosterHandle handle;
};

class PredictWorker : public Nan::AsyncWorker
{
public:
  PredictWorker(Nan::Callback *callback,
                BoosterHandle booster_handle,
                DMatrixHandle mat_handle,
                int mask,
                unsigned ntree)
      : AsyncWorker(callback),
        booster_handle(booster_handle),
        mat_handle(mat_handle),
        mask(mask),
        ntree(ntree),
        out_len(0),
        out_result(nullptr) {}
  ~PredictWorker() {}

  // Executed inside the worker-thread.
  // It is not safe to access V8, or V8 data structures
  // here, so everything we need for input and output
  // should go on `this`.
  void Execute();

  // Executed when the async work is complete
  // this function will be run inside the main event loop
  // so it is safe to use V8 again
  void HandleOKCallback();

private:
  BoosterHandle booster_handle;
  DMatrixHandle mat_handle;
  int mask;
  unsigned ntree;
  bst_ulong out_len;
  const float *out_result;
};

#endif