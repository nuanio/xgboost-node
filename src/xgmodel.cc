#include "xgmodel.h"
#include "xgmatrix.h"

Nan::Persistent<v8::Function> XGModel::constructor;

XGModel::XGModel(BoosterHandle result) : handle(result)
{
}

XGModel::~XGModel()
{
  if (XGBoosterFree(handle) != 0)
  {
    Nan::ThrowTypeError(XGBGetLastError());
  };
  return;
}

void XGModel::Init(v8::Local<v8::Object> exports)
{
  Nan::HandleScope scope;

  // Prepare constructor template
  v8::Local<v8::FunctionTemplate> tpl = Nan::New<v8::FunctionTemplate>(New);
  tpl->SetClassName(Nan::New("CXGModel").ToLocalChecked());
  tpl->InstanceTemplate()->SetInternalFieldCount(1);

  Nan::SetPrototypeMethod(tpl, "predict", Predict);

  constructor.Reset(tpl->GetFunction());
  exports->Set(Nan::New("CXGModel").ToLocalChecked(), tpl->GetFunction());
}

void XGModel::Predict(const Nan::FunctionCallbackInfo<v8::Value> &info)
{
  Nan::EscapableHandleScope scope;
  const size_t MATRIX = 0;
  const size_t MASK = 1;
  const size_t NTREE = 2;

  if (info.Length() != 3 || !info[MATRIX]->IsObject() || !info[MASK]->IsNumber() || !info[NTREE]->IsNumber())
  {
    Nan::ThrowTypeError("Wrong arguments");
    info.GetReturnValue().SetUndefined();
    return;
  }
  XGMatrix *mat = Nan::ObjectWrap::Unwrap<XGMatrix>(info[0]->ToObject());
  XGModel *obj = Nan::ObjectWrap::Unwrap<XGModel>(info.Holder());
  bst_ulong out_len;
  const float *out_result;
  XGBoosterPredict(obj->handle, mat->GetHandle(), info[MASK]->Uint32Value(), info[NTREE]->Uint32Value(), &out_len, &out_result);

  Local<v8::Float32Array> array = Float32Array::New(ArrayBuffer::New(Isolate::GetCurrent(), out_len * sizeof(float)), 0, out_len);
  Nan::TypedArrayContents<float> vfloat(array);
  for (size_t i = 0; i < out_len; i++)
  {
    (*vfloat)[i] = out_result[i];
  }

  info.GetReturnValue().Set(scope.Escape(array));
}

void XGModel::New(const Nan::FunctionCallbackInfo<v8::Value> &info)
{
  Nan::HandleScope scope;
  if (info.IsConstructCall())
  {
    // Invoked as constructor: `new MyObject(...)`
    if (info.Length() != 1 || !info[0]->IsString())
    {
      Nan::ThrowTypeError("Wrong arguments");
      info.GetReturnValue().SetUndefined();
      return;
    }
    BoosterHandle res = nullptr;

    if (XGBoosterCreate(nullptr, 0, &res))
    {
      Nan::ThrowTypeError("Wrong arguments");
      info.GetReturnValue().SetUndefined();
      return;
    }

    auto fname = info[0]->ToString();
    String::Utf8Value value(fname);
    auto cstr = *value ? *value : "<string conversion failed>";
    if (XGBoosterLoadModel(res, cstr))
    {
      Nan::ThrowTypeError(XGBGetLastError());
      info.GetReturnValue().SetUndefined();
      return;
    }
    XGModel *obj = new XGModel(res);
    obj->Wrap(info.This());
    info.GetReturnValue().Set(info.This());
  }
  else
  {
    // Invoked as plain function `MyObject(...)`, turn into construct call.
    size_t argc = info.Length();
    std::unique_ptr<v8::Local<v8::Value>[]> argvp(new v8::Local<v8::Value>[ argc ]);
    auto argv = argvp.get();
    for (size_t i = 0; i != argc; i++)
    {
      argv[i] = info[i];
    }
    v8::Local<v8::Function> cons = Nan::New<v8::Function>(constructor);

    info.GetReturnValue().Set(Nan::NewInstance(cons, argc, argv).ToLocalChecked());
    return;
  }
}
