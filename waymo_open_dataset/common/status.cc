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

#include "waymo_open_dataset/common/status.h"

namespace waymo {
namespace open_dataset {

Status OkStatus() { return Status(); }

Status InvalidArgumentError(const std::string& error_message) {
  return Status(kInvalidArgument, error_message);
}

Status InternalError(const std::string& error_message) {
  return Status(kInternal, error_message);
}

}  // namespace open_dataset
}  // namespace waymo
