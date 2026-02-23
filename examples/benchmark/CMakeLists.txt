
file(GLOB BENCH_SRC benchmark_sift.cpp)

# Benchmark Baseline
add_executable(benchmark_baseline ${BENCH_SRC})
target_link_libraries(benchmark_baseline ezsift_baseline)

# Benchmark Optimized
add_executable(benchmark_optimized ${BENCH_SRC})
target_link_libraries(benchmark_optimized ezsift_opt)
