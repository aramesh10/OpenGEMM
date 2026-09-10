#pragma once

#include <array>

namespace opengemm {

template <auto... Ps> struct List {};

template <auto First, auto... Rest>
constexpr auto to_array(List<First, Rest...>) {
  return std::array<decltype(First), 1 + sizeof...(Rest)>{First, Rest...};
}

}
