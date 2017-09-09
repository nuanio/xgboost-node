#include "xgmatrix.h"
#include "xgmodel.h"

void InitAll(v8::Local<v8::Object> exports)
{
  XGModel::Init(exports);
  XGMatrix::Init(exports);
}

NODE_MODULE(xgboost, InitAll)
