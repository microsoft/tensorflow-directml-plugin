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

#include "numbers.h"

#include <limits>

#include "tfdml/core/util/macros.h"

namespace tfdml
{
namespace strings
{
std::string HumanReadableNumBytes(int64_t num_bytes)
{
    if (num_bytes == std::numeric_limits<int64_t>::min())
    {
        // Special case for number with not representable negation.
        return "-8E";
    }

    const char* neg_str = (num_bytes < 0) ? "-" : "";
    if (num_bytes < 0)
    {
        num_bytes = -num_bytes;
    }

    // Special case for bytes.
    if (num_bytes < 1024)
    {
        // No fractions for bytes.
        char buf[8]; // Longest possible string is '-XXXXB'
        snprintf(
            buf,
            sizeof(buf),
            "%s%lldB",
            neg_str,
            static_cast<long long>(num_bytes));
        return std::string(buf);
    }

    static const char units[] = "KMGTPE"; // int64 only goes up to E.
    const char* unit = units;
    while (num_bytes >= static_cast<int64_t>(1024) * 1024)
    {
        num_bytes /= 1024;
        ++unit;
        CHECK(unit < units + TF_ARRAYSIZE(units));
    }

    // We use SI prefixes.
    char buf[16];
    snprintf(
        buf,
        sizeof(buf),
        ((*unit == 'K') ? "%s%.1f%ciB" : "%s%.2f%ciB"),
        neg_str,
        num_bytes / 1024.0,
        *unit);
    return std::string(buf);
}
} // namespace strings
} // namespace tfdml
