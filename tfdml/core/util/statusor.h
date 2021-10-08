/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "absl/types/optional.h"
#include "tfdml/core/util/macros.h"
#include "tfdml/core/util/status.h"

namespace tfdml {

template <typename T>
class StatusOr {
 public:
  // Constructs a new StatusOr with Status::UNKNOWN status.  This is marked
  // 'explicit' to try to catch cases like 'return {};', where people think
  // StatusOr<std::vector<int>> will be initialized with an empty vector,
  // instead of a Status::UNKNOWN status.
  explicit StatusOr() : status_(Status(TF_UNKNOWN, "")) {}

  // StatusOr<T> will be copy constructible/assignable if T is copy
  // constructible.
  StatusOr(const StatusOr&) = default;
  StatusOr& operator=(const StatusOr&) = default;

  // Conversion copy/move constructor, T must be convertible from U.
  template <typename U, typename std::enable_if<
                            std::is_convertible<U, T>::value>::type* = nullptr>
  StatusOr(const StatusOr<U>& other)
      : status_(other.status_), value_(other.value_) {}

  template <typename U,
            typename std::enable_if<std::is_convertible<U, T>::value>::type*>
  StatusOr(StatusOr<U>&& other) : status_(std::move(other.status())) {
    if (status_.ok()) {
      value_ = std::move(other.ValueOrDie());
    }
  }

  // Conversion copy/move assignment operator, T must be convertible from U.
  template <typename U, typename std::enable_if<
                            std::is_convertible<U, T>::value>::type* = nullptr>
  StatusOr& operator=(const StatusOr<U>& other) {
    if (other.ok()) {
      value_ = other.ValueOrDie();
    }

    status_ = other.status();

    return *this;
  }
  template <typename U, typename std::enable_if<
                            std::is_convertible<U, T>::value>::type* = nullptr>
  StatusOr& operator=(StatusOr<U>&& other);

  // Constructs a new StatusOr with the given value. After calling this
  // constructor, calls to ValueOrDie() will succeed, and calls to status() will
  // return OK.
  //
  // NOTE: Not explicit - we want to use StatusOr<T> as a return type
  // so it is convenient and sensible to be able to do 'return T()'
  // when the return type is StatusOr<T>.
  //
  // REQUIRES: T is copy constructible.
  StatusOr(const T& value) : value_(value) {}

  // Constructs a new StatusOr with the given non-ok status. After calling
  // this constructor, calls to ValueOrDie() will CHECK-fail.
  //
  // NOTE: Not explicit - we want to use StatusOr<T> as a return
  // value, so it is convenient and sensible to be able to do 'return
  // Status()' when the return type is StatusOr<T>.
  //
  // REQUIRES: !status.ok(). This requirement is DCHECKed.
  // In optimized builds, passing Status::OK() here will have the effect
  // of passing tensorflow::error::INTERNAL as a fallback.
  StatusOr& operator=(const Status& status);

  // Similar to the `const T&` overload.
  //
  // REQUIRES: T is move constructible.
  StatusOr(T&& value) : value_(std::move(value)) {}

  // RValue versions of the operations declared above.
  StatusOr(Status&& status) : status_(std::move(status)) {}
  StatusOr& operator=(Status&& status) {
    status_ = std::move(status);
    return *this;
  }

  bool ok() const { return status_.ok(); }

  // Returns a reference to our status. If this contains a T, then
  // returns Status::OK().
  const Status& status() const& { return status_; }

  Status status() && { return std::move(status_); }

  // Returns a reference to our current value, or CHECK-fails if !this->ok().
  //
  // Note: for value types that are cheap to copy, prefer simple code:
  //
  //   T value = statusor.ValueOrDie();
  //
  // Otherwise, if the value type is expensive to copy, but can be left
  // in the StatusOr, simply assign to a reference:
  //
  //   T& value = statusor.ValueOrDie();  // or `const T&`
  //
  const T& ValueOrDie() const& {
    CHECK(status_.ok());
    assert(value_.has_value());
    return *value_;
  }

  T& ValueOrDie() & {
    CHECK(status_.ok());
    assert(value_.has_value());
    return *value_;
  }

  T ConsumeValueOrDie() { return std::move(ValueOrDie()); }

 private:
  absl::optional<T> value_;
  Status status_;
};
}  // namespace tfdml
