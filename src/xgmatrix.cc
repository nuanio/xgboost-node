#include "xgmatrix.h"

Nan::Persistent<v8::Function> XGMatrix::constructor;

XGMatrix::XGMatrix(DMatrixHandle result) : handle(result) {}

XGMatrix::~XGMatrix()
{
  if (XGDMatrixFree(handle) != 0)
  {
    Nan::ThrowTypeError(XGBGetLastError());
  };
  return;
}

void XGMatrix::Init(v8::Local<v8::Object> exports)
{
  Nan::HandleScope scope;

  // Prepare constructor template
  v8::Local<v8::FunctionTemplate> tpl =
      Nan::New<v8::FunctionTemplate>(NewMatrix);

  tpl->SetClassName(Nan::New("CXGMatrix").ToLocalChecked());
  tpl->InstanceTemplate()->SetInternalFieldCount(1);

  Nan::SetPrototypeMethod(tpl, "col", GetCol);
  Nan::SetPrototypeMethod(tpl, "row", GetRow);

  constructor.Reset(tpl->GetFunction());
  exports->Set(Nan::New("CXGMatrix").ToLocalChecked(), tpl->GetFunction());
}

DMatrixHandle XGMatrix::GetHandle() { return handle; }

void XGMatrix::GetCol(const Nan::FunctionCallbackInfo<v8::Value> &info)
{
  Nan::HandleScope scope;

  XGMatrix *obj = Nan::ObjectWrap::Unwrap<XGMatrix>(info.Holder());
  bst_ulong num;
  if (XGDMatrixNumCol(obj->handle, &num))
  {
    Nan::ThrowTypeError(XGBGetLastError());
    info.GetReturnValue().SetUndefined();
    return;
  };
  info.GetReturnValue().Set(Nan::New<Number>(num));
  return;
}

void XGMatrix::GetRow(const Nan::FunctionCallbackInfo<v8::Value> &info)
{
  Nan::HandleScope scope;

  XGMatrix *obj = Nan::ObjectWrap::Unwrap<XGMatrix>(info.Holder());
  bst_ulong num;
  if (XGDMatrixNumRow(obj->handle, &num))
  {
    Nan::ThrowTypeError(XGBGetLastError());
    info.GetReturnValue().SetUndefined();
    return;
  };
  info.GetReturnValue().Set(Nan::New<Number>(num));
  return;
}

void XGMatrix::NewMatrix(const Nan::FunctionCallbackInfo<v8::Value> &info)
{
  Nan::HandleScope scope;
  const size_t INPUT_ARRAY = 0;
  const size_t INPUT_FILE = 1;
  const size_t INPUT_CSC = 2;
  const size_t INPUT_CSR = 3;
  if (info.IsConstructCall())
  {
    DMatrixHandle res = nullptr;
    // 0: dense, 1: file
    if (info.Length() > 0 || !info[0]->IsNumber() ||
        info[0]->Uint32Value() > 4)
    {
      switch (info[0]->Uint32Value())
      {
      case INPUT_ARRAY:
      {
        if (XGMatrix::FromDense(info, res))
        {
          info.GetReturnValue().SetUndefined();
          return;
        }
        break;
      };
      case INPUT_FILE:
      {
        if (XGMatrix::FromFile(info, res))
        {
          info.GetReturnValue().SetUndefined();
          return;
        }
        break;
      };
      case INPUT_CSC:
      {
        if (XGMatrix::FromCSCR(info, res, true))
        {
          info.GetReturnValue().SetUndefined();
          return;
        }
        break;
      };
      case INPUT_CSR:
      {
        if (XGMatrix::FromCSCR(info, res, false))
        {
          info.GetReturnValue().SetUndefined();
          return;
        }
        break;
      };
      default:
      {
        Nan::ThrowTypeError("matrix type is not supported");
        info.GetReturnValue().SetUndefined();
        return;
      }
      }
    }
    else
    {
      Nan::ThrowTypeError("empty argument");
      info.GetReturnValue().SetUndefined();
      return;
    }

    XGMatrix *obj = new XGMatrix(res);
    obj->Wrap(info.This());
    info.GetReturnValue().Set(info.This());
    return;
  }
  else
  {
    // Invoked as plain function `MyObject(...)`, turn into construct call.
    size_t argc = info.Length();
    std::unique_ptr<v8::Local<v8::Value>[]> argvp(new v8::Local<v8::Value>[argc]);
    auto argv = argvp.get();
    for (size_t i = 0; i != argc; i++)
    {
      argv[i] = info[i];
    }
    v8::Local<v8::Function> cons = Nan::New<v8::Function>(constructor);

    info.GetReturnValue().Set(
        Nan::NewInstance(cons, argc, argv).ToLocalChecked());
    return;
  }
}

int XGMatrix::FromCSCR(const Nan::FunctionCallbackInfo<v8::Value> &info,
                       DMatrixHandle &res, bool column)
{
  const size_t DATA = 1;
  const size_t INDPTR = 2;
  const size_t INDICES = 3;
  const size_t NINDPTR = 4;
  const size_t NINDICES = 5;
  const size_t NNUM = 6;
  Nan::HandleScope scope;
  if (info.Length() != 7
      || !info[DATA]->IsFloat32Array()
      || !info[INDPTR]->IsUint32Array()
      || !info[INDICES]->IsUint32Array()
      || !info[NINDPTR]->IsNumber()
      || !info[NINDICES]->IsNumber()
      || !info[NNUM]->IsNumber())
  {
    Nan::ThrowTypeError(
        "Wrong arguments: this function takes 4 arguments"
        "'data : Float32Array', 'indptr : Uint32Array',"
        "'indices : Uint32Array', 'n : Int'");
    return 1;
  }
  auto nindptr = info[NINDPTR]->Uint32Value();
  auto nelem = info[NINDICES]->Uint32Value();
  auto num_row = info[NNUM]->Uint32Value();

  Local<TypedArray> colptr = info[INDPTR].As<TypedArray>();
  Nan::TypedArrayContents<uint32_t> vcolptr(colptr);

  std::unique_ptr<size_t[]> vcolptr_size_t(new size_t[nindptr]);
  auto vcolptr_p = vcolptr_size_t.get();
  if (*vcolptr != NULL)
  {
    for (size_t i = 0; i != nindptr; i++)
    {
      vcolptr_p[i] = (*vcolptr)[i];
    }
  }
  else
  {
    Nan::ThrowTypeError(
        "indptr is an invalid Uint32Array");
    return 1;
  }

  Local<TypedArray> jsArray = info[DATA].As<TypedArray>();
  Nan::TypedArrayContents<float> vfloat(jsArray);

  Local<TypedArray> indices = info[INDICES].As<TypedArray>();
  Nan::TypedArrayContents<uint32_t> vindices(indices);

  if (*vfloat != NULL && *vindices != NULL)
  {
    bool err;

    if (column)
    {
      err = XGDMatrixCreateFromCSCEx(vcolptr_p, *vindices, *vfloat, nindptr,
                                     nelem, num_row, &res);
    }
    else
    {
      err = XGDMatrixCreateFromCSREx(vcolptr_p, *vindices, *vfloat, nindptr,
                                     nelem, num_row, &res);
    }

    if (err)
    {
      Nan::ThrowTypeError(XGBGetLastError());
      return 1;
    }
  }
  else
  {
    Nan::ThrowTypeError("data should be a Float32Array, indices should be a Uint32Array");
    return 1;
  }

  return 0;
}

int XGMatrix::FromDense(const Nan::FunctionCallbackInfo<v8::Value> &info,
                        DMatrixHandle &res)
{
  Nan::HandleScope scope;
  const size_t DATA = 1;
  const size_t ROW = 2;
  const size_t COL = 3;
  const size_t MISSING = 4;
  if (info.Length() != 5 || !info[DATA]->IsFloat32Array() ||
      !info[ROW]->IsNumber() || !info[COL]->IsNumber() || !info[MISSING]->IsNumber())
  {
    Nan::ThrowTypeError(
        "Wrong arguments: fromDense takes 4 arguments 'array : Float32Array', "
        "'row: Int', 'col: Int', 'missing: float'");
    return 1;
  }
  Local<TypedArray> jsArray = info[DATA].As<TypedArray>();
  Nan::TypedArrayContents<float> vfloat(jsArray);
  if (*vfloat != NULL)
  {
    auto rownum = info[ROW]->Uint32Value();
    auto colnum = info[COL]->Uint32Value();
    // for (size_t i = 0; i < rownum; i++) {
    //   for (size_t j = 0; j < colnum; j++) {
    //       printf("%8.6f ", (*vfloat)[i*colnum + j]);
    //   }
    //   printf("\n");
    // }
    if (XGDMatrixCreateFromMat(*vfloat, rownum, colnum, info[MISSING]->NumberValue(),
                               &res))
    {
      Nan::ThrowTypeError(XGBGetLastError());
      return 1;
    }
  }
  else
  {
    Nan::ThrowTypeError("array should be a Float32Array");
    return 1;
  }

  return 0;
}

int XGMatrix::FromFile(const Nan::FunctionCallbackInfo<v8::Value> &info,
                       DMatrixHandle &res)
{
  Nan::HandleScope scope;

  if (info.Length() != 2 || !info[1]->IsString())
  {
    Nan::ThrowTypeError(
        "Wrong arguments: fromFile takes one argument 'file: String'");
    return 1;
  }
  auto fname = info[1]->ToString();
  String::Utf8Value value(fname);
  auto cstr = *value ? *value : "<string conversion failed>";
  try {
    if (XGDMatrixCreateFromFile(cstr, 1, &res))
    {
      Nan::ThrowTypeError(XGBGetLastError());
      return 1;
    }
  } catch (dmlc::Error& e) {
    Nan::ThrowTypeError(e.what());
    return 1;
  }

  return 0;
}
