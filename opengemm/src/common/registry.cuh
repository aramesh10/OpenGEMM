#pragma once

#include <array>
#include <cstdint>

namespace opengemm {

template <auto... Ps> struct List {};

template <int COLS, class Row, auto... Ps>
constexpr auto make_table(Row row, List<Ps...>) {
  std::array<int32_t, sizeof...(Ps) * COLS> table{};
  int at = 0;
  for (const auto &values : {row(Ps)...})
    for (int32_t value : values) table[at++] = value;
  return table;
}

constexpr int name_count(const char *text) {
  int names = 1;
  for (const char *at = text; *at; ++at)
    if (*at == ',') ++names;
  return names;
}

}
