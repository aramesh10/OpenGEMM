// The compile-time table both builds are, minus what is in it. Each
// src/<impl>/registry.cuh names its kernels as a List; each launch.cuh turns
// that one pack into the rows a caller reads and the launchers it dispatches
// on, so row i of the registry is kernel i of the launcher table.
#pragma once

#include <array>
#include <cstdint>

namespace opengemm {

// The kernels a build compiles, as a type. The policies are non-type template
// arguments, so `List` takes `auto`: the dense and the block-scaled Policy are
// different types and this holds either.
template <auto... Ps> struct List {};

// The registry as flat int32, COLS per policy. `row` spells one policy in the
// harness vocabulary, which is the build's own business; this only lays the
// rows end to end, folded at compile time so the table costs the build
// nothing.
template <int COLS, class Row, auto... Ps>
constexpr auto make_table(Row row, List<Ps...>) {
  std::array<int32_t, sizeof...(Ps) * COLS> table{};
  int at = 0;
  for (const auto &values : {row(Ps)...})
    for (int32_t value : values) table[at++] = value;
  return table;
}

// The count of commas plus one, so a field name added without a column (or the
// reverse) is a build error rather than a misread table.
constexpr int name_count(const char *text) {
  int names = 1;
  for (const char *at = text; *at; ++at)
    if (*at == ',') ++names;
  return names;
}

}  // namespace opengemm
