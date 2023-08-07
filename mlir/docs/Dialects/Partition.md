# 'partition' Dialect

---
*Note:* At this point this is mostly a pre-pre-alpha design doc. Please let me
know the aspects needs revision.

---

This dialect provides the concept of `DistTensor`. The `DistTensor`, a
distributed tensor, can be thought as a `SparseTensor` possibly residing in
different memory spaces. When accessing slices of `DistTensor` often special
handling is needed.

## Proposal

### Attributes and Enums

We add `distAttribute` as an additional encoding to `Tensor`. This optional
encoding is present only if the sparsity encoding is also present[^1]. The
attribute is a list of enum elements of size equal to the number of dimensions
in the SparseTensor. The enum values are `replicated`, `distributed` denoting if
the dimension is `replicated`, present with in nodes or `distributed`
(uniformly[^2]) over a grid dimension.

### Operations

The operations are ordered by the life-cycle of a `DistTensor` object. The
object of `DistTensor` is not created by any op and is assumed to be originating
as function argument. The creation and destruction are assumed to be handled by
caller of the function being optimized by the dialect, henceforth referred as
the function.

#### `partion.get_grid_size`

The operation returns an array denoting the extents of the grid on which the
function will run. The array size (rank of grid) needs to be known at the
compile time and should match at the run-time, otherwise it's undefined behavior.
(Might or might not be checked by dialect at runtime).

#### `partion.get_slice`

```
partion.get_slice(%distTensor: DistTensor, %my_processor_id: List[int],
  %gridSize: List[int], %slice_id: List[int]) -> SparseTensor
```

Given `%distTensor` handle, the operation checks if `%my_processor_id` has the
requested slice denoted by `%slice_id` and returns corresponding `SparseTensor`.
If the slice is not present, the operation will do the necessary communication
to acquire the slice. At least one processor is expected to have that slice id
which provides it, otherwise it's an undefined behavior.

The `%gridSize` argument is necessary to compute the slice owner of `%slice_id`.

## limitations

1. `DistTensor` is spread out mostly in the memories of distinct cluster nodes.
   e.g. mpi/accelerators with multiple compute nodes.
   
1. Only one of the dense dimension, is allowed to be `distributed`. With
   assumption of it having distributed along fastest varying dimension of
   processor grid (assuming row-major ordering), for convenience.

1. There is no `partition.set_slice`, a node is responsible for computing an
   output slice and is owner of that slice. Other nodes can call `get_slice`
   including the owner in subsequent operations but the ownership of slice never
   changes which would require `set_slice` operation.

## Open Questions

1) Is `dimension` or `level` has `distAttribute`? \\
  `SparseTensors` have concept
   of `Dimensions`, tensor dimensions (an abstract concept) and `Levels`
   denoting levels in sparse tensor tree structure. Different levels are marked
   as `Compressed`/`Dense` etc. not dimensions. For both CSR and CSC, outer
   level is dense and the inner level sparse. They are distinguished using
   `dimToLevel` affine map which either maps `(i, j) -> (i, j)` for CSR and `(i,
   j) -> (j, i)` for CSC. So do we want to say partition encoding denotes if
   each `level` is replicated or distributed?
      
[^1]: We are more interested in sparsity in the current projects and that way we
    keep the phase ordering linear. `DistTensor` $\rightarrow$ `SparseTensor`
    $\rightarrow$ `(Vanilla)Tensor` (linalg).

[^2]: Each node getting equal points in that dimension.
