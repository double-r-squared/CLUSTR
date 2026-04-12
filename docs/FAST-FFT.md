# Implementing a fast parrellel multidemensional FFT using advanced MPI

## 1. Conceptual Foundation

The algorithm performs a **global redistribution** of a multidimensional array distributed across processes. It eliminates local data rearrangement by using **descriptors** of non-contiguous memory regions and a **generalized all-to-all** communication primitive.

**Key Idea:** Instead of moving data locally to make it contiguous before sending, we describe to the communication library exactly which scattered pieces to gather and where to place them.

---

## 2. Prerequisites (Language-Agnostic)

| Capability | Description |
|------------|-------------|
| **MPI-2 or later** | Access to `MPI_Type_create_subarray` and `MPI_Alltoallw`. |
| **Serial FFT** | A function that computes a 1D FFT along a specified contiguous axis of a local array. |
| **Cartesian Process Topology** | Ability to arrange processes into a multidimensional grid and extract 1D subgroups along each dimension. |
| **Array Layout** | Row-major (C-style) or column-major (Fortran-style) must be consistent; the algorithm requires knowing which axis is contiguous in memory. |

---

## 3. The Core Redistribution Routine

This routine swaps the distributed axis of an array from axis `v` to axis `w` within a **1D process group**.

### Inputs (Generic)

- `comm`: A communicator representing a 1D group of processes.
- `element_type`: The MPI datatype of a single array element.
- `ndims`: Number of dimensions.
- `global_shape`: Array of length `ndims` with the total size in each dimension.
- `v`: Current distributed axis (0 ≤ v < ndims).
- `w`: Target distributed axis (0 ≤ w < ndims).
- `input_array`: Local buffer whose shape is `global_shape` but with dimension `v` replaced by its local size on `comm`.
- `output_array`: Local buffer whose shape is `global_shape` but with dimension `w` replaced by its local size on `comm`.

### Steps (Algorithmic Description)

#### Step 1: Compute Local Partition Sizes and Offsets

For each process rank `p` in `comm`, compute:
- `send_size[p]`: Number of elements along axis `v` that will be sent to rank `p`.
- `send_start[p]`: Starting index along axis `v` for the chunk destined to rank `p`.
- `recv_size[p]`: Number of elements along axis `w` that will be received from rank `p`.
- `recv_start[p]`: Starting index along axis `w` for the chunk coming from rank `p`.

These are computed using a **balanced block decomposition**:
```
base = floor(global_shape[axis] / group_size)
remainder = global_shape[axis] mod group_size
if rank < remainder:
    size = base + 1
    start = size * rank
else:
    size = base
    start = size * rank + remainder
```

#### Step 2: Create Descriptors for Scattered Data Regions

For each process `p` in `comm`:

**Send Descriptor:**
- Describes a subarray of the **global** array shape.
- The subarray has the same size in all dimensions except axis `v`, where its size is `send_size[p]`.
- Its starting position in the global array is zero everywhere except along axis `v`, where it is `send_start[p]`.
- This descriptor tells the system: "From my local input buffer, the data destined for rank `p` is scattered according to this subarray pattern."

**Receive Descriptor:**
- Similarly describes a subarray of the **global** array shape.
- The subarray has size `recv_size[p]` along axis `w`, and full global size along other axes.
- Its start is zero except along axis `w`, where it is `recv_start[p]`.
- This descriptor tells the system: "Place incoming data from rank `p` into my local output buffer according to this scattered pattern."

These descriptors are created using the equivalent of `MPI_Type_create_subarray`.

#### Step 3: Perform Generalized All-to-All Exchange

Call the generalized all-to-all communication routine (`MPI_Alltoallw`).

For each destination `p`:
- Send: 1 instance of the send descriptor from `input_array` (with displacement 0).
- Receive: 1 instance of the receive descriptor into `output_array` (with displacement 0).

The communication library uses the descriptors to pack, transfer, and unpack data without any explicit local rearrangement.

#### Step 4: Release Descriptors

Free the created datatype resources.

---

## 4. Building the Full Multidimensional FFT

### 4.1 Slab Decomposition (1D Process Grid)

1. **Initial State:** Array distributed along axis 0.
2. **Local Transforms:** Perform 1D FFTs along all axes **except** axis 0 (these are fully local).
3. **Redistribute:** Call core routine to swap axis 0 ↔ axis 1.
4. **Final Transform:** Perform 1D FFTs along axis 0 (now local).
5. **Inverse:** Reverse steps for backward transform.

### 4.2 Pencil Decomposition (2D Process Grid)

Assume a 3D array and a 2D process grid with subgroups `P0` (rows) and `P1` (columns).

**Forward Transform Sequence:**

| Step | Operation | Communicator | Axes State |
|------|-----------|--------------|------------|
| 1 | Local FFT along axis 2 | none | distributed in 0 and 1, whole in 2 |
| 2 | Redistribute axis 2 → axis 1 | `P1` | distributed in 0 and 2, whole in 1 |
| 3 | Local FFT along axis 1 | none | distributed in 0 and 2, whole in 1 |
| 4 | Redistribute axis 1 → axis 0 | `P0` | distributed in 1 and 2, whole in 0 |
| 5 | Local FFT along axis 0 | none | fully transformed |

### 4.3 Higher-Dimensional Generalization

For a `d`-dimensional array and a `(d-1)`-dimensional process grid:
- Create `d-1` 1D subgroups.
- Alternate between **local FFT** on the non-distributed axis and **redistribution** on the appropriate subgroup to bring the next axis into local alignment.

---

## 5. Implementation Considerations (Language-Agnostic)

### 5.1 Memory Strategy
- The redistribution is **out-of-place**: you need separate input and output buffers.
- For a complete FFT, you may need multiple buffer shapes. Pre-allocate all intermediate array shapes at initialization to avoid runtime overhead.

### 5.2 Planning and Reuse
- Descriptors depend only on global shape, process grid, and axis pair.
- **Precompute all descriptors once** and reuse for every transform of the same problem size.
- This matches the paper's approach: they measured performance over 50 repeated transforms using the same precomputed plans.

### 5.3 Serial FFT Integration
- Your serial FFT routine must be able to operate on a **contiguous axis** of a multidimensional array (i.e., with a given stride between elements of the same transform).
- Many libraries (FFTW, MKL, cuFFT) provide "many" or "batch" interfaces for this.

### 5.4 MPI Environment
- The algorithm relies on `MPI_Alltoallw`. Performance may vary by MPI implementation.
- No vendor-specific optimizations are required; the method's advantage comes from eliminating local memory copies.

---

## 6. Pseudocode (Language-Agnostic)

```
function redistribute(comm, element_type, ndims, global_shape, v, w, input, output):
    group_size = comm.size
    my_rank = comm.rank

    // Step 1: Compute partitions
    send_sizes = array[group_size]
    send_starts = array[group_size]
    recv_sizes = array[group_size]
    recv_starts = array[group_size]

    for p in 0..group_size-1:
        send_sizes[p], send_starts[p] = decompose(global_shape[v], group_size, p)
        recv_sizes[p], recv_starts[p] = decompose(global_shape[w], group_size, p)

    // Step 2: Create descriptors
    send_types = array[group_size]
    recv_types = array[group_size]

    for p in 0..group_size-1:
        // Subshape for send: full except at axis v
        subshape_send = copy(global_shape)
        subshape_send[v] = send_sizes[p]
        starts_send = array[ndims] filled with 0
        starts_send[v] = send_starts[p]
        send_types[p] = create_subarray(ndims, global_shape, subshape_send, starts_send, element_type)

        // Subshape for receive: full except at axis w
        subshape_recv = copy(global_shape)
        subshape_recv[w] = recv_sizes[p]
        starts_recv = array[ndims] filled with 0
        starts_recv[w] = recv_starts[p]
        recv_types[p] = create_subarray(ndims, global_shape, subshape_recv, starts_recv, element_type)

    // Step 3: Communicate
    send_counts = array[group_size] filled with 1
    send_displs = array[group_size] filled with 0
    recv_counts = array[group_size] filled with 1
    recv_displs = array[group_size] filled with 0

    alltoallw(input, send_counts, send_displs, send_types,
              output, recv_counts, recv_displs, recv_types,
              comm)

    // Step 4: Cleanup
    for p in 0..group_size-1:
        free_type(send_types[p])
        free_type(recv_types[p])
```

---

## 7. Summary: The Essence of the Algorithm

| Traditional Approach | This Paper's Approach |
|----------------------|----------------------|
| 1. Local transpose to pack data contiguously. | 1. Describe scattered data with subarray descriptors. |
| 2. `MPI_Alltoallv` on contiguous buffers. | 2. Single `MPI_Alltoallw` call using descriptors. |
| 3. (Sometimes) local unpacking transpose. | 3. Data arrives already in correct scattered layout. |

**Core Benefit:** The algorithm pushes the complexity of non-contiguous data movement into the MPI library, potentially leveraging hardware offload, and eliminates CPU-intensive local transposes.