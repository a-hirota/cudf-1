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
#include <cuda/iterator>
#include <cuda/std/execution>
#include <cuda/std/iterator>
#include <cuda/stream>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/remove.h>
#include <thrust/transform_reduce.h>

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
 * @brief Upper bound on the segment size handled by the CUB fast path
 *
 * cub::DeviceBatchedTopK needs a compile-time maximum for its segment-size argument, and
 * a looser bound costs more temporary storage. Segments larger than this take the
 * sort-based path instead.
 */
constexpr size_type cub_max_segment_size = 1024;

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

  auto segment_sizes = cuda::args::deferred_sequence{
    cudf::detail::make_counting_transform_iterator(
      0,
      [d_offsets] __device__(size_type i) -> size_type { return d_offsets[i + 1] - d_offsets[i]; }),
    cuda::args::static_bounds<0, cub_max_segment_size>{}};

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

  // CUB requires us to accept unsorted output, but the sort-based path returns each
  // segment's top k in order and callers rely on that. Restore it by sorting only the k
  // selected rows of each segment: O(k log k) per segment instead of the O(n log n) full
  // segmented sort the fast path just avoided. This is also the shape CUB itself intends
  // for ordered output -- select first, then sort the selection.
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

struct dispatch_segmented_topk_fn {
  column_view col;
  column_view segment_offsets;
  size_type k;
  order topk_order;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  template <typename T>
    requires(cudf::is_fixed_width<T>() and !cudf::is_floating_point<T>() and !cudf::is_chrono<T>())
  std::unique_ptr<column> operator()()
  {
    return cub_segmented_top_k_order<T>(col, segment_offsets, k, topk_order, stream, mr);
  }

  template <typename T>
    requires(cudf::is_chrono<T>())
  std::unique_ptr<column> operator()()
  {
    using rep_type = typename T::rep;
    return cub_segmented_top_k_order<rep_type>(col, segment_offsets, k, topk_order, stream, mr);
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
  // only whether every segment is within [k, cub_max_segment_size].
  auto const d_offsets = segment_offsets.begin<size_type>();
  return thrust::transform_reduce(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    thrust::counting_iterator<size_type>(0),
    thrust::counting_iterator<size_type>(num_segments),
    [d_offsets, k] __device__(size_type i) -> bool {
      auto const size = d_offsets[i + 1] - d_offsets[i];
      return size >= k && size <= cub_max_segment_size;
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

  if (can_use_cub_batched_topk(col, segment_offsets, k, stream)) {
    return type_dispatcher<dispatch_storage_type>(
      col.type(), dispatch_segmented_topk_fn{col, segment_offsets, k, topk_order, stream, mr});
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
