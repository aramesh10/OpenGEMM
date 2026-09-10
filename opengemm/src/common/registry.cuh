#pragma once

#include <array>
#include <type_traits>

namespace opengemm {

template <auto... Ps> struct List {};

template <auto First, auto... Rest>
constexpr auto to_array(List<First, Rest...>) {
  return std::array<std::remove_cvref_t<decltype(First)>, 1 + sizeof...(Rest)>{
      First, Rest...};
}

}
