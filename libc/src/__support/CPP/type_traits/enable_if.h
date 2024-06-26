//===-- enable_if type_traits -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC_SUPPORT_CPP_TYPE_TRAITS_ENABLE_IF_H
#define LLVM_LIBC_SRC_SUPPORT_CPP_TYPE_TRAITS_ENABLE_IF_H

#include "src/__support/CPP/type_traits/type_identity.h"

namespace __llvm_libc::cpp {

// enable_if
template <bool B, typename T> struct enable_if;
template <typename T> struct enable_if<true, T> : type_identity<T> {};
template <bool B, typename T = void>
using enable_if_t = typename enable_if<B, T>::type;

} // namespace __llvm_libc::cpp

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_TYPE_TRAITS_ENABLE_IF_H
