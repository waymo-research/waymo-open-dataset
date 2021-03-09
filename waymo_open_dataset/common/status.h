/* Copyright 2021 The Waymo Open Dataset Authors. All Rights Reserved.

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

#ifndef WAYMO_OPEN_DATASET_COMMON_STATUS_H_
#define WAYMO_OPEN_DATASET_COMMON_STATUS_H_

#include <string>

namespace waymo {
namespace open_dataset {

enum StatusCode : int { kOk = 0, kInvalidArgument = 1, kInternal = 13 };

// Simple class to track error and error messages.
class Status {
 public:
  Status() { _code = kOk; }

  Status(StatusCode code, const std::string& error_message) {
    _code = code;
    _error_message = error_message;
  }

  inline bool ok() const { return _code == kOk; }

  const std::string& message() const { return _error_message; }

  const StatusCode status_code() const { return _code; }

 private:
  StatusCode _code;
  std::string _error_message;
};

Status OkStatus();

Status InvalidArgumentError(const std::string& error_message);

Status InternalError(const std::string& error_message);

}  // namespace open_dataset
}  // namespace waymo

#endif  // WAYMO_OPEN_DATASET_COMMON_STATUS_H_
