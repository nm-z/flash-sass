#include <sycl/sycl.hpp>
#include <cstdio>
int main() {
  sycl::queue q{sycl::gpu_selector_v};
  std::printf("device %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
  float *a = sycl::malloc_device<float>(1 << 20, q);
  q.parallel_for(sycl::range<1>(1 << 20), [=](sycl::id<1> i) { a[i] = 2.0f * i[0]; }).wait();
  float h;
  q.memcpy(&h, a + 1000, 4).wait();
  std::printf("acpp a[1000]=%f\n", h);
  sycl::free(a, q);
}
