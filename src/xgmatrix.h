#ifndef XGMAT_H
#define XGMAT_H

#include "base.h"

class XGMatrix : public Nan::ObjectWrap
{
public:
  static void Init(v8::Local<v8::Object> exports);
  DMatrixHandle GetHandle();

private:
  explicit XGMatrix(DMatrixHandle result);
  ~XGMatrix();

  static NAN_METHOD(NewMatrix);

  static int FromDense(const Nan::FunctionCallbackInfo<v8::Value> &info, DMatrixHandle &res);
  static int FromCSCR(const Nan::FunctionCallbackInfo<v8::Value> &info, DMatrixHandle &res, bool C);
  static int FromFile(const Nan::FunctionCallbackInfo<v8::Value> &info, DMatrixHandle &res);
  static NAN_METHOD(GetCol);
  static NAN_METHOD(GetRow);
  static Nan::Persistent<v8::Function> constructor;
  DMatrixHandle handle;
};

#endif