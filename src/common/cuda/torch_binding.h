#pragma once

#include <torch/extension.h>

#define STRINGFY(str) #str

#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
    m.def(STRINGFY(func), &func, STRINGFY(func));
