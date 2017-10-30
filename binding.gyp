{
  "targets": [
    {
      "target_name": "xgboost",
      "sources": [
        "src/index.cc",
        "src/xgmodel.cc",
        "src/xgmatrix.cc"
      ],
      "include_dirs": [
        "<!(node -e \"require('nan')\")",
        "./xgboost/include/xgboost",
        "./xgboost/dmlc-core/include",
        "./xgboost/rabit/include"
      ],
      "cflags!": [ "-fno-exceptions" ],
      "cflags_cc!": [ "-fno-exceptions" ],
      "ccflags": [
        "-g",
        "-std=c++11",
        "-Wall",
        "-Wextra",
        "-fexceptions"
      ],
      "variables": {
        "OMP": "<!(cat ./xgboost/rabit/lib/flag)"
      },
      "libraries": [
        "../xgboost/lib/libxgboost.a",
        "../xgboost/rabit/lib/librabit_empty.a",
        "../xgboost/dmlc-core/libdmlc.a",
        "<(OMP)"
      ],
      "configurations": {
        "Release": {
          "msvs_settings": {
            "VCCLCompilerTool": {
              "ExceptionHandling": "1"
            }
          }
        }
      },
      "conditions": [
        [
          "OS!='win'",
          {
            "cflags+": [
              "-std=c++11"
            ],
            "cflags_c+": [
              "-std=c++11"
            ],
            "cflags_cc+": [
              "-std=c++11"
            ],
          }
        ],
        [
          "OS == 'mac'",
          {
            "xcode_settings": {
              "CLANG_CXX_LIBRARY": "libc++",
              "WARNING_CFLAGS": [
                "-Wall",
                "-Wextra",
                "-Wno-unused-parameter"
              ],
              'GCC_ENABLE_CPP_EXCEPTIONS': 'YES',
              'GCC_ENABLE_CPP_RTTI': 'YES',
            }
          }
        ]
      ]
    }
  ]
}
