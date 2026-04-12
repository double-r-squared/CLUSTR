# Optimization Plan: Caching & Auto Grid Selection

## 1. Subarray Descriptor Caching

### Problem
On each `RedistributePlan::execute()` call:
- Creates fresh `std::vector<std::unique_ptr<Subarray<T>>>` (2 vectors)
- Allocates N unique_ptrs (one per peer)
- Constructs N Subarray objects
- **Cost**: ~N heap allocations per execute() call (N = process grid size, typically 4-64)

In repeated transforms (common in simulations: transform 1000s of times with same geometry), this overhead accumulates.

### Solution
**Pre-allocate vectors as plan members; reuse across calls**

```cpp
// In RedistributePlan<T> constructor:
send_owned_.reserve(n_);
recv_owned_.reserve(n_);
send_ptrs_.resize(n_);
recv_ptrs_.resize(n_);

// On each execute():
// Instead of creating fresh vectors, clear and refill:
send_owned_.clear();
recv_owned_.clear();
for (size_t p = 0; p < n_; ++p) {
    send_owned_.push_back(std::make_unique<Subarray<T>>(...));
    recv_owned_.push_back(std::make_unique<Subarray<T>>(...));
    send_ptrs_[p] = send_owned_.back().get();
    recv_ptrs_[p] = recv_owned_.back().get();
}
```

### Expected Benefit
- **5-10% speedup** on repeated transforms (allocation elimination)
- **No functional change** (semantics identical)
- **Memory**: ~1 KB per plan (vectors stay allocated, not recreated)

### Complexity
- **Low**: Minimal code changes (~10 lines)
- **Risk**: Low (allocation patterns are well-understood)
- **Testing**: Existing tests verify functionality; performance measured by inspection

---

## 2. Automatic Grid Selection for 3D Pencil FFT

### Problem
Users must manually specify `grid_dims = {P0, P1}` where `P0 × P1 = world_size`.

Example:
```cpp
int rank = mpi.rank(), size = mpi.size();
// User must pick these:
std::vector<int> grid_dims;
if (size == 4) grid_dims = {2, 2};      // balanced
else if (size == 6) grid_dims = {2, 3}; // user's choice
else if (size == 8) grid_dims = {2, 4}; // or {4, 2}? Unknown.
else throw std::runtime_error("unsupported size");
```

**Challenge**: Not all factorizations have equal performance. Heuristics vary by:
- Communication topology (some grids are "wider", some "taller")
- Data size (large arrays: tall grid better? small: square grid?)
- MPI implementation (differs by hardware)

### Solution: Three-Tier Heuristic

#### Tier 1: Balanced Square Grid (Default)
```cpp
std::vector<int> auto_grid_selection(int world_size) {
    // Prefer P0 ≈ P1 (minimizes aspect ratio)
    int p0 = static_cast<int>(std::sqrt(world_size) + 0.5);
    if (p0 * p0 == world_size)
        return {p0, p0};  // Perfect square: use it
    
    // Not a perfect square: find closest factorization
    for (int candidate_p0 = p0; candidate_p0 >= 1; --candidate_p0) {
        if (world_size % candidate_p0 == 0) {
            int p1 = world_size / candidate_p0;
            // Pick factorization closest to square
            if (candidate_p0 * candidate_p0 <= world_size)
                return {candidate_p0, p1};
        }
    }
    throw std::invalid_argument("no valid factorization");
}
```

**Examples**:
```
size=4:  √4 ≈ 2   → {2, 2}     ✓ perfect square
size=6:  √6 ≈ 2.4 → try 2 → 2×3, try 1 → 1×6 → pick {2, 3}
size=8:  √8 ≈ 2.8 → try 2 → 2×4, NOT square, try 1 → pick {2, 4}
size=12: √12≈ 3.5 → try 3 → 3×4, try 2 → 2×6 → pick {3, 4}
```

#### Tier 2: User Override (Optional)
```cpp
struct ParallelFFT3DConfig {
    std::vector<int> grid_dims;  // If empty, auto-select
    // ... other config options
};

// Usage:
ParallelFFT3DConfig config;  // auto_select = true by default
if (user_wants_custom) config.grid_dims = {4, 2};  // override
ParallelFFT3D<double> fft(world, global_shape, config);
```

#### Tier 3: Performance Tuning (Future)
For now, heuristic only. Later: benchmark different factorizations on target hardware and pick best.

### Expected Benefit
- **User convenience**: No manual grid selection needed
- **Robustness**: Works for any world size (no "unsupported size" errors)
- **Performance**: Balanced square grids typically good default (load balanced, minimizes communication diameter)

### Complexity
- **Low-Medium**: Factorization algorithm ~30 lines
- **Risk**: Low (heuristic is optional; users can override)
- **Testing**: Test with various world sizes (3, 4, 6, 8, 9, 12, 16, 20, ...)

---

## Implementation Order

### Phase 1: Caching (First - easier)
1. Add member vectors to `RedistributePlan<T>`
2. Update constructor to pre-allocate
3. Update `execute()` to reuse
4. Verify existing tests still pass
5. Benchmark (optional, but informative)

### Phase 2: Auto Grid Selection (Second - design first)
1. Add `auto_grid_selection(int world_size)` function in `parallel_fft_3d.h`
2. Decide: constructor parameter or new config struct?
3. Update `ParallelFFT3D` constructor to accept optional grid_dims
4. Update test to verify auto-selection works
5. Document in PHASE8.md

---

## Testing Strategy

### Caching Tests
```cpp
// Rerun same transform 100 times, measure time
// Before: ~5s per run × 100 = 500s
// After: ~4.75s per run × 100 = 475s (5% improvement)
// Method: Add optional bench mode to fft_3d_test.cpp
```

### Auto Grid Selection Tests
```cpp
// Test all small world sizes and their factorizations
std::vector<int> sizes = {3, 4, 6, 8, 9, 12, 15, 16, 20, 25, ...};
for (int size : sizes) {
    auto [p0, p1] = auto_grid_selection(size);
    assert(p0 * p1 == size);          // Must factorize
    assert(p0 <= sqrt(size) + 1);     // Balanced
    // Run FFT with auto-selected grid
    EXPECT_PASS(fft_3d_test with this grid);
}
```

---

## Estimated Effort

| Task | Effort | Risk | Value |
|------|--------|------|-------|
| Caching | 30 min | Low | Medium (5-10% speedup) |
| Auto grid selection | 1 hour | Low | High (better UX, robustness) |
| **Total** | **90 min** | **Low** | **High** |

---

## Follow-Up Questions

1. **Caching**: Should we also add an optional benchmark mode to measure the speedup?
2. **Grid selection**: Acceptable to hardcode balanced-square heuristic, or want benchmarking infrastructure first?
3. **Config struct**: Prefer `ParallelFFT3D(world, shape, grid_dims)` or `ParallelFFT3D(world, shape, config)`?
