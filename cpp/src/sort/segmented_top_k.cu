/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sort.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sequence.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_batched_topk.cuh>
#include <cub/device/device_topk.cuh>
#include <cuda/iterator>
#include <cuda/std/execution>
#include <cuda/std/iterator>
#include <cuda/stream>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <vector>

namespace cudf {
namespace detail {
namespace {

/**
 * @brief Resolves the k indices per segment
 *
 * Marks values outside the k range to -1 to be removed in a separate step.
 * Rows not covered by any segment are also marked to be removed.
 * Also computes the total number of valid indices for each segment.
 * All elements are used in a segment if it has less than k total elements.
 *
 * @param d_offsets Offsets for each segment
 * @param k Number of values to keep in each segment
 * @param d_indices Mark these indices to be removed
 * @param d_segment_sizes Store actual sizes of each segment
 */
CUDF_KERNEL void resolve_segment_indices(device_span<size_type const> d_offsets,
                                         size_type k,
                                         device_span<size_type> d_indices,
                                         size_type* d_segment_sizes)
{
  auto const tid = cudf::detail::grid_1d::global_thread_id();
  if (tid >= d_indices.size()) { return; }

  auto const sitr = thrust::upper_bound(thrust::seq, d_offsets.begin(), d_offsets.end(), tid);
  // Mark rows outside all segments for removal (offsets need not cover all rows).
  if (sitr == d_offsets.begin() || sitr == d_offsets.end()) {
    d_indices[tid] = -1;
    return;
  }
  auto const segment_start = *(sitr - 1);
  auto const segment_end   = *sitr;
  auto const index         = tid - segment_start;
  if (index >= k) { d_indices[tid] = -1; }  // mark values outside of top k

  if (index == 0) {
    auto const segment_size  = segment_end - segment_start;
    auto const segment_index = cuda::std::distance(d_offsets.begin(), sitr) - 1;
    // segment is k or less elements
    d_segment_sizes[segment_index] = cuda::std::min(k, segment_size);
  }
}

/**
 * @brief Computes the top k indices per segment by fully sorting each segment
 *
 * Sorts every segment and then drops the indices beyond the first k of each one.
 * This is the general fallback: it supports all types, nulls, and any segment size.
 *
 * @param col Column to compute the top k indices for
 * @param segment_offsets Offsets for each segment
 * @param k Number of indices to keep per segment
 * @param topk_order Whether the top k are the largest or the smallest values
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Lists column of the top k indices for each segment
 */
std::unique_ptr<column> sort_based_segmented_top_k_order(column_view const& col,
                                                         column_view const& segment_offsets,
                                                         size_type k,
                                                         order topk_order,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  auto const size_data_type = data_type{type_to_id<size_type>()};

  auto const nulls   = topk_order == order::ASCENDING ? null_order::AFTER : null_order::BEFORE;
  auto const temp_mr = cudf::get_current_device_resource_ref();
  auto const indices = cudf::detail::segmented_sorted_order(
    cudf::table_view({col}), segment_offsets, {topk_order}, {nulls}, stream, temp_mr);
  auto const d_indices = indices->mutable_view().begin<size_type>();

  // Zero-initialized because resolve_segment_indices writes a segment's size only from its
  // first element; an empty segment has none, so its slot must remain 0, not uninitialized.
  auto segment_sizes = cudf::detail::make_zeroed_device_uvector_async<size_type>(
    segment_offsets.size() - 1, stream, temp_mr);
  auto span_indices = device_span<size_type>{d_indices, static_cast<std::size_t>(indices->size())};
  auto const grid   = cudf::detail::grid_1d(indices->size(), 256);
  resolve_segment_indices<<<grid.num_blocks, grid.num_threads_per_block, 0, stream>>>(
    segment_offsets, k, span_indices, segment_sizes.data());
  CUDF_CUDA_TRY(cudaGetLastError());
  auto [offsets, total_elements] =
    cudf::detail::make_offsets_child_column(segment_sizes.begin(), segment_sizes.end(), stream, mr);

  auto result = cudf::make_fixed_width_column(
    size_data_type, total_elements, mask_state::UNALLOCATED, stream, mr);
  auto d_result = result->mutable_view().begin<size_type>();
  // remove the indices marked by resolve_segment_indices
  thrust::remove_copy(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      d_indices,
                      d_indices + indices->size(),
                      d_result,
                      -1);

  auto const num_rows = static_cast<size_type>(offsets->size() - 1);
  return make_lists_column(
    num_rows, std::move(offsets), std::move(result), 0, rmm::device_buffer{});
}

/**
 * @brief Upper bound on the segment size handled by the batched CUB fast path
 *
 * cub::DeviceBatchedTopK's baseline backend processes one segment per thread block and
 * rejects, at compile time, any statically-known maximum no policy can cover within the
 * shared-memory limit; 2048 is the largest value that compiles for every type
 * instantiated here.
 *
 * A looser bound also costs more temporary storage, but measuring the two candidates
 * showed the wider bound is worth it: raising 1024 to 2048 slowed 1024-row segments by
 * roughly a tenth while making 2048-row segments, which would otherwise fall back to a
 * full sort, about two orders of magnitude faster.
 *
 * CUB versions carrying the thread-block-cluster backend (CCCL PR 9224) route segments
 * beyond the baseline bound to that backend instead of rejecting them, raising the
 * per-segment limit to 2^21 rows -- but only on devices with clusters (SM 9.0+), which
 * cub_max_segment_size_for_device() checks at run time.
 */
#if __has_include(<cub/agent/agent_batched_topk_cluster.cuh>)
constexpr size_type cub_max_segment_size = size_type{1} << 21;
constexpr bool cub_has_cluster_backend   = true;
#else
constexpr size_type cub_max_segment_size = 2048;
constexpr bool cub_has_cluster_backend   = false;
#endif
constexpr size_type cub_baseline_max_segment_size = 2048;

/**
 * @brief Returns the largest segment the batched CUB path may take on this device
 */
size_type cub_max_segment_size_for_device()
{
  if constexpr (cub_has_cluster_backend) {
    int device = 0;
    CUDF_CUDA_TRY(cudaGetDevice(&device));
    int cluster_launch = 0;
    CUDF_CUDA_TRY(cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch, device));
    if (cluster_launch == 0) { return cub_baseline_max_segment_size; }
  }
  return cub_max_segment_size;
}

/**
 * @brief Returns true if the column's type is supported by the CUB fast path
 *
 * Mirrors the condition used by the non-segmented top_k in top_k.cu: no nulls, and a
 * fixed-width non-floating-point type (floating point needs special NaN handling).
 */
bool is_fast_path(column_view const& column)
{
  return !column.has_nulls() && cudf::is_fixed_width(column.type()) &&
         !cudf::is_floating_point(column.type());
}

/**
 * @brief Sorts the selected indices of each list by their values and builds the result
 *
 * CUB requires us to accept unsorted output, but the sort-based path returns each
 * segment's top k in order and callers rely on that. Restore it by sorting only the
 * selected rows of each segment: O(k log k) per segment instead of the O(n log n) full
 * segmented sort the fast path avoided. This is also the shape CUB itself intends for
 * ordered output -- select first, then sort the selection.
 */
std::unique_ptr<column> sorted_top_k_lists(std::unique_ptr<column> offsets,
                                           std::unique_ptr<column> child,
                                           column_view const& col,
                                           order topk_order,
                                           size_type num_segments,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  auto const nulls   = topk_order == order::ASCENDING ? null_order::AFTER : null_order::BEFORE;
  auto const temp_mr = cudf::get_current_device_resource_ref();
  auto const keys    = cudf::detail::gather(cudf::table_view({col}),
                                         child->view(),
                                         out_of_bounds_policy::DONT_CHECK,
                                         negative_index_policy::NOT_ALLOWED,
                                         stream,
                                         temp_mr);
  auto sorted        = cudf::detail::stable_segmented_sort_by_key(cudf::table_view({child->view()}),
                                                           keys->view(),
                                                           offsets->view(),
                                                                  {topk_order},
                                                                  {nulls},
                                                           stream,
                                                           mr);

  return make_lists_column(num_segments,
                           std::move(offsets),
                           std::move(sorted->release().front()),
                           0,
                           rmm::device_buffer{});
}

/**
 * @brief Computes the top k indices per segment using cub::DeviceBatchedTopK
 *
 * Selects the top k of each segment without sorting it, and writes the global row
 * indices of the selected rows. Every segment emits exactly k indices, so the caller
 * must have verified that no segment is smaller than k.
 *
 * The output within a segment is unordered and may be non-deterministic, which the CUB
 * API requires us to acknowledge through the execution environment.
 */
template <typename T>
std::unique_ptr<column> cub_segmented_top_k_order(column_view const& col,
                                                  column_view const& segment_offsets,
                                                  size_type k,
                                                  order topk_order,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  auto const num_segments = segment_offsets.size() - 1;
  auto const d_offsets    = segment_offsets.begin<size_type>();

  auto indices =
    rmm::device_uvector<size_type>(static_cast<std::size_t>(num_segments) * k, stream, mr);
  auto d_out    = indices.data();
  auto const in = col.begin<T>();

  // cub::DeviceBatchedTopK takes iterators over per-segment iterators. The segments have
  // different lengths, so these cannot be strided iterators as in CUB's own uniform-size
  // example; each one resolves segment i through the offsets column.
  auto keys_in = cudf::detail::make_counting_transform_iterator(
    0, [in, d_offsets] __device__(size_type i) -> T const* { return in + d_offsets[i]; });
  auto keys_out = cuda::make_constant_iterator(cuda::make_discard_iterator());
  // Seeding each segment's values with its start offset makes the selected values the
  // global row indices, which is what the lists child column stores.
  auto values_in = cudf::detail::make_counting_transform_iterator(
    0, [d_offsets] __device__(size_type i) -> cuda::counting_iterator<size_type> {
      return cuda::counting_iterator<size_type>{d_offsets[i]};
    });
  auto values_out = cudf::detail::make_counting_transform_iterator(
    0, [d_out, k] __device__(size_type i) -> size_type* {
      return d_out + static_cast<std::size_t>(i) * k;
    });

  // Materialized rather than a transform iterator: newer CUB requires the
  // deferred_sequence to wrap a random-access iterator over integral values, which a
  // device pointer satisfies on every CCCL version we build against.
  auto sizes = rmm::device_uvector<size_type>(num_segments, stream);
  thrust::transform(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    thrust::counting_iterator<size_type>(0),
    thrust::counting_iterator<size_type>(num_segments),
    sizes.begin(),
    [d_offsets] __device__(size_type i) -> size_type { return d_offsets[i + 1] - d_offsets[i]; });
  auto segment_sizes = cuda::args::deferred_sequence{
    sizes.data(), cuda::args::static_bounds<0, cub_max_segment_size>{}};

  auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                               cuda::execution::tie_break::unspecified,
                                               cuda::execution::output_ordering::unsorted);
  auto env          = cuda::std::execution::env{cuda::stream_ref{stream.value()}, requirements};

  auto tmp_size = std::size_t{0};
  if (topk_order == order::ASCENDING) {
    CUDF_CUDA_TRY(cub::DeviceBatchedTopK::MinPairs(nullptr,
                                                   tmp_size,
                                                   keys_in,
                                                   keys_out,
                                                   values_in,
                                                   values_out,
                                                   segment_sizes,
                                                   k,
                                                   num_segments,
                                                   env));
    auto tmp = rmm::device_buffer(tmp_size, stream);
    CUDF_CUDA_TRY(cub::DeviceBatchedTopK::MinPairs(tmp.data(),
                                                   tmp_size,
                                                   keys_in,
                                                   keys_out,
                                                   values_in,
                                                   values_out,
                                                   segment_sizes,
                                                   k,
                                                   num_segments,
                                                   env));
  } else {
    CUDF_CUDA_TRY(cub::DeviceBatchedTopK::MaxPairs(nullptr,
                                                   tmp_size,
                                                   keys_in,
                                                   keys_out,
                                                   values_in,
                                                   values_out,
                                                   segment_sizes,
                                                   k,
                                                   num_segments,
                                                   env));
    auto tmp = rmm::device_buffer(tmp_size, stream);
    CUDF_CUDA_TRY(cub::DeviceBatchedTopK::MaxPairs(tmp.data(),
                                                   tmp_size,
                                                   keys_in,
                                                   keys_out,
                                                   values_in,
                                                   values_out,
                                                   segment_sizes,
                                                   k,
                                                   num_segments,
                                                   env));
  }

  // Every segment produced exactly k indices, so the offsets are a simple multiple of k.
  auto offsets = cudf::detail::sequence(
    num_segments + 1,
    numeric_scalar<size_type>(0, true, stream, cudf::get_current_device_resource_ref()),
    numeric_scalar<size_type>(k, true, stream, cudf::get_current_device_resource_ref()),
    stream,
    mr);
  auto child = std::make_unique<column>(std::move(indices), rmm::device_buffer{}, 0);

  return sorted_top_k_lists(
    std::move(offsets), std::move(child), col, topk_order, num_segments, stream, mr);
}

/**
 * @brief Upper bound on the segment count handled by the per-segment CUB path
 *
 * Each segment costs one host-driven cub::DeviceTopK launch, so this path only pays off
 * while the per-launch overhead stays negligible next to the selection work itself. The
 * motivating shape is a handful of huge partitions (TPC-DS Q67 has about ten); beyond
 * this many segments the sort-based path takes over.
 */
constexpr size_type cub_topk_loop_max_segments = 64;

/**
 * @brief Computes the top k indices per segment with one cub::DeviceTopK call per segment
 *
 * Complements the batched path: cub::DeviceBatchedTopK processes one segment per thread
 * block and therefore cannot exceed cub_max_segment_size rows, while a non-segmented
 * cub::DeviceTopK call uses the whole GPU for one segment of any size. With few segments
 * the loop of full-device selections is far cheaper than fully sorting every segment.
 *
 * Segments holding k or fewer rows contribute all of their row indices, matching the
 * sort-based path's min(k, segment_size) output shape.
 */
template <typename T>
std::unique_ptr<column> cub_per_segment_top_k_order(column_view const& col,
                                                    column_view const& segment_offsets,
                                                    size_type k,
                                                    order topk_order,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  auto const num_segments = segment_offsets.size() - 1;
  auto const h_offsets    = cudf::detail::make_host_vector(
    device_span<size_type const>{segment_offsets.begin<size_type>(),
                                    static_cast<std::size_t>(num_segments) + 1},
    stream);

  auto h_out_offsets = std::vector<size_type>(num_segments + 1);
  h_out_offsets[0]   = 0;
  for (size_type i = 0; i < num_segments; ++i) {
    auto const size      = h_offsets[i + 1] - h_offsets[i];
    h_out_offsets[i + 1] = h_out_offsets[i] + cuda::std::min(size, k);
  }
  auto const total = h_out_offsets[num_segments];

  auto const temp_mr = cudf::get_current_device_resource_ref();
  auto indices       = rmm::device_uvector<size_type>(total, stream, temp_mr);
  auto const in = col.begin<T>();
  auto keys_out = cuda::make_discard_iterator();

  auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                               cuda::execution::output_ordering::unsorted);
  auto env          = cuda::std::execution::env{cuda::stream_ref{stream.value()}, requirements};

  auto run = [&](void* tmp, std::size_t& tmp_size, size_type i) {
    auto const begin    = h_offsets[i];
    auto const size     = h_offsets[i + 1] - begin;
    auto const keys_in  = in + begin;
    auto const vals_in  = cuda::counting_iterator<size_type>{begin};
    auto const vals_out = indices.data() + h_out_offsets[i];
    return topk_order == order::ASCENDING
             ? cub::DeviceTopK::MinPairs(
                 tmp, tmp_size, keys_in, keys_out, vals_in, vals_out, size, k, env)
             : cub::DeviceTopK::MaxPairs(
                 tmp, tmp_size, keys_in, keys_out, vals_in, vals_out, size, k, env);
  };

  auto max_tmp_size = std::size_t{0};
  for (size_type i = 0; i < num_segments; ++i) {
    if (h_offsets[i + 1] - h_offsets[i] <= k) { continue; }
    auto tmp_size = std::size_t{0};
    CUDF_CUDA_TRY(run(nullptr, tmp_size, i));
    max_tmp_size = cuda::std::max(max_tmp_size, tmp_size);
  }
  auto tmp = rmm::device_buffer(max_tmp_size, stream);

  for (size_type i = 0; i < num_segments; ++i) {
    auto const size = h_offsets[i + 1] - h_offsets[i];
    if (size <= 0) { continue; }
    if (size <= k) {
      // The whole segment is selected; its row indices need no CUB call.
      thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                       indices.begin() + h_out_offsets[i],
                       indices.begin() + h_out_offsets[i + 1],
                       h_offsets[i]);
      continue;
    }
    auto tmp_size = max_tmp_size;
    CUDF_CUDA_TRY(run(tmp.data(), tmp_size, i));
  }

  // Synchronous copy: h_out_offsets is stack-local and the async copy defers the read.
  auto offsets = std::make_unique<column>(
    cudf::detail::make_device_uvector(h_out_offsets, stream, mr), rmm::device_buffer{}, 0);
  auto child = std::make_unique<column>(std::move(indices), rmm::device_buffer{}, 0);

  return sorted_top_k_lists(
    std::move(offsets), std::move(child), col, topk_order, num_segments, stream, mr);
}

struct dispatch_segmented_topk_fn {
  column_view col;
  column_view segment_offsets;
  size_type k;
  order topk_order;
  bool batched;  // one thread block per segment vs. one full-device selection per segment
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  template <typename T>
    requires(cudf::is_fixed_width<T>() and !cudf::is_floating_point<T>() and !cudf::is_chrono<T>())
  std::unique_ptr<column> operator()()
  {
    return batched
             ? cub_segmented_top_k_order<T>(col, segment_offsets, k, topk_order, stream, mr)
             : cub_per_segment_top_k_order<T>(col, segment_offsets, k, topk_order, stream, mr);
  }

  template <typename T>
    requires(cudf::is_chrono<T>())
  std::unique_ptr<column> operator()()
  {
    using rep_type = typename T::rep;
    return this->template operator()<rep_type>();
  }

  template <typename T>
    requires(not cudf::is_fixed_width<T>() or cudf::is_floating_point<T>())
  std::unique_ptr<column> operator()()
  {
    CUDF_UNREACHABLE("unexpected type for segmented_top_k fast path");
  }
};

/**
 * @brief Returns true if cub::DeviceBatchedTopK can be used for these segments
 *
 * Beyond the type/null requirements, the CUB path needs every segment to fit the
 * compile-time size bound, and needs no segment to be smaller than k, since it emits
 * exactly k indices per segment while the sort path emits min(k, segment_size).
 */
bool can_use_cub_batched_topk(column_view const& col,
                              column_view const& segment_offsets,
                              size_type k,
                              rmm::cuda_stream_view stream)
{
  if (!is_fast_path(col)) { return false; }

  auto const num_segments = segment_offsets.size() - 1;
  if (num_segments <= 0) { return false; }

  // Reduce straight to the answer in a single pass: the sizes themselves are not needed,
  // only whether every segment is within [k, the largest this device's backends take].
  auto const max_size  = cub_max_segment_size_for_device();
  auto const d_offsets = segment_offsets.begin<size_type>();
  return thrust::transform_reduce(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    thrust::counting_iterator<size_type>(0),
    thrust::counting_iterator<size_type>(num_segments),
    [d_offsets, k, max_size] __device__(size_type i) -> bool {
      auto const size = d_offsets[i + 1] - d_offsets[i];
      return size >= k && size <= max_size;
    },
    true,
    thrust::logical_and<bool>{});
}
}  // namespace

std::unique_ptr<column> segmented_top_k_order(column_view const& col,
                                              column_view const& segment_offsets,
                                              size_type k,
                                              order topk_order,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(k >= 0, "k must be greater than or equal to 0", std::invalid_argument);

  auto const size_data_type = data_type{type_to_id<size_type>()};
  if (k == 0 || col.is_empty()) { return cudf::make_empty_lists_column(size_data_type); }

  CUDF_EXPECTS(segment_offsets.size() > 0,
               "segment_offsets must have at least one element",
               std::invalid_argument);

  CUDF_EXPECTS(segment_offsets.type() == size_data_type,
               "segment_offsets must be of type INT32",
               cudf::data_type_error);
  CUDF_EXPECTS(segment_offsets.null_count() == 0,
               "segment_offsets must not have nulls",
               std::invalid_argument);

  if (is_fast_path(col)) {
    if (can_use_cub_batched_topk(col, segment_offsets, k, stream)) {
      return type_dispatcher<dispatch_storage_type>(
        col.type(),
        dispatch_segmented_topk_fn{col, segment_offsets, k, topk_order, true, stream, mr});
    }
    // Few segments that the batched path rejected (they are too large, or smaller than
    // k): select each one with a full-device cub::DeviceTopK call instead.
    if (auto const num_segments = segment_offsets.size() - 1;
        num_segments > 0 && num_segments <= cub_topk_loop_max_segments) {
      return type_dispatcher<dispatch_storage_type>(
        col.type(),
        dispatch_segmented_topk_fn{col, segment_offsets, k, topk_order, false, stream, mr});
    }
  }

  return sort_based_segmented_top_k_order(col, segment_offsets, k, topk_order, stream, mr);
}

std::unique_ptr<column> segmented_top_k(column_view const& col,
                                        column_view const& segment_offsets,
                                        size_type k,
                                        order topk_order,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  if (col.is_empty()) { return cudf::make_empty_column(col.type()); }

  auto ordered =
    cudf::detail::segmented_top_k_order(col, segment_offsets, k, topk_order, stream, mr);
  auto lv = cudf::lists_column_view(ordered->view());
  if (lv.is_empty()) { return cudf::make_empty_lists_column(col.type()); }

  auto result         = cudf::detail::gather(cudf::table_view({col}),
                                     lv.child(),
                                     out_of_bounds_policy::DONT_CHECK,
                                     negative_index_policy::NOT_ALLOWED,
                                     stream,
                                     mr);
  auto offsets        = std::move(ordered->release().children.front());
  auto const num_rows = static_cast<size_type>(offsets->size() - 1);
  return make_lists_column(
    num_rows, std::move(offsets), std::move(result->release().front()), 0, rmm::device_buffer{});
}

}  // namespace detail

std::unique_ptr<column> segmented_top_k(column_view const& col,
                                        column_view const& segment_offsets,
                                        size_type k,
                                        order topk_order,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_top_k(col, segment_offsets, k, topk_order, stream, mr);
}

std::unique_ptr<column> segmented_top_k_order(column_view const& col,
                                              column_view const& segment_offsets,
                                              size_type k,
                                              order topk_order,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_top_k_order(col, segment_offsets, k, topk_order, stream, mr);
}
}  // namespace cudf
