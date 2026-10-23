#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>
int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<float *> x("x", 1 << 20);
    Kokkos::parallel_for(x.extent(0), KOKKOS_LAMBDA(int i) { x(i) = 2.0f * i; });
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
    std::printf("kokkos exec=%s x[1000]=%f\n", Kokkos::DefaultExecutionSpace::name(), h(1000));
    Kokkos::print_configuration(std::cout, false);
  }
  Kokkos::finalize();
}
