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
  static Nan::Persistent<v8::Function> constructor;
  BoosterHandle handle;
};

#endif