# Unifying 2D Slab and 3D Pencil FFTs: Architecture Analysis

## Current State

### Phase 7: ParallelFFT2D (Slab Decomposition)
```
Process Grid:    1D (linear array of P processes)
Decomposition:   Slab (one axis distributed at a time)
Array Shape:     2D: [N0, N1]
Distribution:    Alternates: axis 0 ↔ axis 1
Steps:           2 steps (1 FFT + 1 redistribute + 1 FFT)
Intermediate:    Single scratch buffer (DistArray with scalar distributed_axis)
```

**Structure**:
```cpp
class ParallelFFT2D<T> {
    Comm comm_;                           // 1D process group
    RedistributePlan<T> fwd_plan_;        // Single redistribution plan
    RedistributePlan<T> inv_plan_;
    // No explicit scratch buffer (uses input as scratch, then output)
};
```

### Phase 8: ParallelFFT3D (Pencil Decomposition)
```
Process Grid:    2D (P0 × P1 processes)
Decomposition:   Pencil (two axes distributed simultaneously)
Array Shape:     3D: [N0, N1, N2]
Distribution:    Complex: P0 on axis 0, P1 on axes 1↔2
Steps:           5 steps (FFT + redist + FFT + redist + FFT)
Intermediate:    Four intermediate DistArray views (dual-axis complexity)
```

**Structure**:
```cpp
class ParallelFFT3D<T> {
    std::optional<Comm> p0_comm_, p1_comm_;  // 2D sub-communicators
    std::optional<RedistributePlan<T>> p1_fwd_plan_, p0_fwd_plan_;
    std::optional<RedistributePlan<T>> p0_inv_plan_, p1_inv_plan_;
    std::optional<DistArray<T>> scratch_b_p1_view_, scratch_b_p0_view_;  // Dual views
    std::optional<DistArray<T>> scratch_b_p0_inv_view_, scratch_b_p1_inv_view_;
};
```

---

## Proposed Unified Framework

### Conceptual Design: `ParallelFFT<N, Decomposition>`

```cpp
template <typename T, int Ndim, DecompositionStrategy Strategy>
class ParallelFFT {
    // Generic N-dimensional FFT with pluggable decomposition
    // Strategy = SLAB | PENCIL | BLOCK | ...
};

// Specializations:
template <typename T> using ParallelFFT2D_Slab = ParallelFFT<T, 2, SLAB>;
template <typename T> using ParallelFFT3D_Pencil = ParallelFFT<T, 3, PENCIL>;
template <typename T> using ParallelFFT3D_Slab = ParallelFFT<T, 3, SLAB>;  // NEW
```

---

## Detailed Pro/Con Analysis

### PROS: Unification

#### 1. **Code Reuse**
- **Description**: Common machinery for FFT orchestration, normalization, validation
- **Benefit**: Reduce ~600 lines of duplicate code (constructor pattern, forward/inverse mirroring, error handling)
- **Example shared**:
  - Balanced block decomposition (already in redistribute.h)
  - Normalization factor computation
  - Validation logic (shape matching, axis bounds)
  - Barrier synchronization patterns
- **Quantified**: ~40% code reduction in the core FFT classes

#### 2. **Extensibility**
- **Description**: Adding higher dimensions becomes incremental
- **Current**: 2D → 3D required ~400 new lines, extensive copy-paste
- **Unified**: 4D, 5D FFTs would fit the same template pattern
- **Benefit**: Support arbitrary N-dimensional FFTs with minimal new code
- **Business value**: Future domains (4D simulations, higher-order tensors) unlock without refactoring

#### 3. **Consistent API**
- **Description**: All FFTs expose same interface
- **Current**:
  ```cpp
  ParallelFFT2D<double> fft2d(comm, {N0, N1});
  co_await fft2d.forward(input, output);
  
  ParallelFFT3D<double> fft3d(world, {N0, N1, N2}, grid_dims);
  co_await fft3d.forward(input, output);  // Different constructor signature!
  ```
- **Unified**:
  ```cpp
  ParallelFFT<double, 2, SLAB> fft2d(comm, {N0, N1});
  ParallelFFT<double, 3, PENCIL> fft3d(comm_2d, {N0, N1, N2}, grid_dims);
  // Same forward/inverse interface, constructor varies by strategy
  ```
- **Benefit**: Easier learning curve, template metaprogramming (same code paths for 2D/3D)

#### 4. **Unified Testing & Validation**
- **Description**: Parametrized test suite validates all dimensions/strategies uniformly
- **Current**: Separate test files (fft_2d_test.cpp, fft_3d_test.cpp) with different logic
- **Unified**:
  ```cpp
  template <int Ndim, Strategy S>
  void run_fft_test(const std::vector<size_t>& shape, ...) {
      // Single test implementation for all Ndim × Strategy combinations
  }
  // Instantiate: 2D slab, 3D pencil, 3D slab, 4D pencil, etc.
  ```
- **Benefit**: Catch regressions uniformly; easier to add new dimension/strategy combinations

#### 5. **Maintainability**
- **Description**: Bug fixes apply universally
- **Current**: If a normalize computation bug is found, must fix in both 2D and 3D classes
- **Unified**: Fix once in shared base template
- **Example**: If `inv_norm_axis` computation is discovered to have precision issues, unified version fixes all dimensions

#### 6. **Documentation Alignment**
- **Description**: Single conceptual framework described once, not twice
- **Current**: FAST-FFT.md describes algorithm abstractly; implementation split across 2D/3D classes
- **Unified**: Generic docs describe parameterized algorithm; specializations inherit naturally

---

### CONS: Unification

#### 1. **Template Complexity** ⚠️ **MAJOR**
- **Description**: Generic template instantiation becomes complex
- **Current**:
  ```cpp
  ParallelFFT2D<double> fft(comm, {64, 64});  // Simple, concrete
  ```
- **Unified**:
  ```cpp
  // Option A: Explicit template parameters (verbose)
  ParallelFFT<double, 2, DecompositionStrategy::SLAB> fft(comm, {64, 64});
  
  // Option B: Template deduction (requires C++20 concepts, CTAD)
  // Impossible to deduce all three T, Ndim, Strategy from arguments alone
  // Would need helpers:
  auto fft = make_fft<SLAB>(comm, shape);  // Still unclear about Ndim
  ```
- **Impact**: Users need more template knowledge; harder to teach/use
- **Compiler**: Significantly longer compile times (template specialization explosion)
  - 2D + 3D = ~100 instantiations (T × Ndim × Strategy × Transport × Recv combinations)
  - Linker bloat (duplicate code for each specialization)

#### 2. **Compilation Performance** ⚠️ **MAJOR**
- **Description**: Generic templates expand to N × M combinations
- **Current**:
  - 2D: ~2 specializations (float, double)
  - 3D: ~2 specializations (float, double)
  - Total: ~4 instantiations
- **Unified**:
  - Ndim: 2, 3, 4, 5, ... = N choices
  - Strategy: SLAB, PENCIL, BLOCK, ... = M choices
  - Total: **N × M × 2 specializations** (for float/double)
  - If N=3, M=3: **18 instantiations** (9x growth)
- **Compile time**: Expected 5-10x longer for full test suite
- **Mitigation options**:
  - Explicit instantiation in translation units (*.cpp)
  - Template factoring (move common code to non-template base)
  - Lazy instantiation (only compile used combinations)

#### 3. **Cognitive Load & Documentation**
- **Description**: Users must understand parameterization logic
- **Current**:
  ```cpp
  // Clear: slab = 1D grid, one redistribution
  ParallelFFT2D<double> fft(comm, {N0, N1});
  
  // Clear: pencil = 2D grid, two redistributions
  ParallelFFT3D<double> fft(world, {N0, N1, N2}, {{P0, P1}});
  ```
- **Unified**:
  ```cpp
  // Which is slab? Which is pencil? What does Ndim=3 + SLAB mean?
  ParallelFFT<double, 3, SLAB> fft3d_slab(...);  // 3D slab (1D grid, axis alternates)
  ParallelFFT<double, 3, PENCIL> fft3d_pencil(...);  // 3D pencil (2D grid)
  ```
- **Documentation burden**: Must explain strategy interactions with dimensionality
- **Error messages**: Template instantiation failures are cryptic

#### 4. **Incremental Mismatch**
- **Description**: Not all dimension × strategy combinations are valid/useful
- **Current**: Only implemented what's needed (2D slab, 3D pencil)
- **Unified temptation**: Implement 3D slab, 4D pencil, 4D slab, ...
- **Problem**: Not all combinations have the same performance/complexity
  - 2D slab: simple (1 redistribute)
  - 3D slab: complex (2 redistributes, alternating axes)
  - 3D pencil: manageable (2 redistributes, fixed axes per sub-comm)
  - 4D: ???
- **Maintenance**: Would be responsible for proving correctness/performance of combinations not yet deployed

#### 5. **Specialization Logic Complexity**
- **Description**: Some strategies require different constructor signatures, member arrangements
- **Example**:
  - **Slab 2D**: `ParallelFFT2D(Comm comm, shape)`
    - 1D process grid, simple API
  - **Pencil 3D**: `ParallelFFT3D(Comm world, shape, grid_dims)`
    - 2D process grid, requires grid specification
  - **Slab 3D** (if we add it): `ParallelFFT3D_Slab(Comm comm, shape)`?
    - Back to 1D grid, but 3D array — different from pencil
- **Unified**: Constructor must be flexible:
  ```cpp
  // Generic version needs to handle both 1D and 2D sub-communicators
  template <int Ndim, Strategy S>
  ParallelFFT<T, Ndim, S> {
      // For SLAB: Comm (1D)
      // For PENCIL: Comm (2D derived via cart_create)
      // Template specialization required for each strategy
  };
  ```
- **Impact**: Lose clarity gained by separate classes

#### 6. **Memory Overhead (Unused Features)**
- **Description**: Unified class carries members for all strategies, but only uses subset
- **Current**:
  - ParallelFFT2D: single `comm_`, two `RedistributePlan`
  - ParallelFFT3D: two `Comm`, four `RedistributePlan`, four `DistArray` views
- **Unified**:
  ```cpp
  // Must accommodate max complexity
  std::optional<Comm> comm_;                    // For SLAB
  std::optional<Comm> p0_comm_, p1_comm_;       // For PENCIL
  std::vector<std::optional<RedistributePlan<T>>> plans;  // Variable count
  ```
- **Impact**: Small (~few KB per object), but symbol table bloat at compile time

#### 7. **Testing Explosion**
- **Description**: Test coverage for all combinations
- **Current**: 2 test files (2D, 3D), each ~200 lines
- **Unified**: Must test all Ndim × Strategy × grid_size combinations
  - 2D slab: grids 1, 3, 4, 6, ...
  - 3D pencil: grids 4, 6, 8, ...
  - 3D slab: grids 1, 3, 4, 6, ...
  - 4D?: ...
- **Maintenance**: Each new dimension requires new grid configurations, reference signals

---

## Comparison Matrix

| Aspect | Current (Separate) | Unified | Winner |
|--------|-------------------|---------|--------|
| **Code Duplication** | ~600 lines | ~400 lines | Unified (40% reduction) |
| **Compile Time** | ~5s | ~30-50s | Current (10x faster) |
| **Template Complexity** | Simple | High (C++20 concepts?) | Current (easier to understand) |
| **Adding 4D FFT** | New class, ~500 lines | Template specialization, ~100 lines | Unified (50% less code) |
| **API Consistency** | Inconsistent (different constructors) | Consistent | Unified |
| **Testability** | Duplicated logic, hard to verify | Parametrized tests | Unified (easier coverage) |
| **Maintainability** | Bug fixes in 2 places | Bug fixes in 1 place | Unified |
| **Binary Size** | ~2 MB | ~4-5 MB (multi-specialization) | Current (50% smaller) |
| **User Learning Curve** | Learn both, separate APIs | Learn one pattern, but complex | Current (simpler) |
| **Extensibility** | Requires new class | Automatic via template | Unified |

---

## Recommendation: Staged Approach

### Stage 1 (Current): Keep Separate ✓ **DO THIS FIRST**
- **Timeline**: Now
- **Action**: Leave ParallelFFT2D and ParallelFFT3D as independent classes
- **Rationale**:
  - Prototyping phase; unclear what patterns generalize
  - Compilation performance matters for dev iteration
  - API stability: changing to unified breaks user code
  - Low immediate maintenance burden (only 2 implementations)

### Stage 2 (After 4D): Extract Common Base
- **Trigger**: When 3D slab or 4D FFT proposed
- **Action**: Extract shared code into non-templated base:
  ```cpp
  class ParallelFFTBase {
      // Shared: normalization, validation, planning
      void compute_inverse_norms(...);
      void validate_shapes(...);
  };
  
  template <typename T> class ParallelFFT2D : ParallelFFTBase { ... };
  template <typename T> class ParallelFFT3D : ParallelFFTBase { ... };
  ```
- **Benefit**: Reuse without template explosion
- **Cost**: ~50 lines of factoring, minimal compile impact

### Stage 3 (If Pattern Matures): Full Unification
- **Trigger**: 4+ implementations show clear pattern, users request unified API
- **Action**: Convert to unified template with CTAD helpers
  ```cpp
  // Explicit specialization only for deployed combinations
  template class ParallelFFT<double, 2, SLAB>;    // Force instantiation
  template class ParallelFFT<double, 3, PENCIL>;
  template class ParallelFFT<double, 3, SLAB>;
  // ... only what's actually used
  ```
- **Benefit**: Unified API, extensible for new dimensions/strategies
- **Cost**: Template complexity, but mitigated by early explicit instantiation

---

## Detailed Comparison: 3D Slab as Test Case

To illustrate the unification complexity, let's consider adding **3D slab decomposition** (3D array on 1D process grid):

### Current (Separate) Approach
```cpp
class ParallelFFT3D_Slab {
    Comm comm_;                    // 1D process group
    RedistributePlan<T> plan_10_;  // axis 1 ↔ axis 0
    RedistributePlan<T> plan_21_;  // axis 2 ↔ axis 1
    std::optional<DistArray<T>> scratch_ab_;
    // ... ~300 lines, follows 2D slab pattern
};

// Usage:
ParallelFFT3D_Slab<double> fft(comm, {N0, N1, N2});
co_await fft.forward(input, output);
```

**Pros**: Clear, isolated implementation; inherits 2D pattern

**Cons**: Yet another class; similar to pencil but different enough that code doesn't share

### Unified Approach
```cpp
template <int Ndim, Strategy S> class ParallelFFT<T, Ndim, S> {
    // General template
};

// Specialization for 3D slab:
template <typename T> class ParallelFFT<T, 3, Strategy::SLAB> {
    Comm comm_;
    std::vector<RedistributePlan<T>> plans;  // Variable count: Ndim-1 plans
    std::vector<std::optional<DistArray<T>>> scratches;  // Variable count
    // Generic loop:
    // for each axis except first:
    //   FFT (axis i)
    //   Redistribute (axis i ↔ i-1)
};

// Usage: same as before, but now parametric
auto fft = make_fft<3, SLAB>(comm, {N0, N1, N2});
co_await fft.forward(input, output);
```

**Pros**: Shared implementation for all slab FFTs (2D, 3D, 4D, ...)

**Cons**: Generic loop requires care; harder to reason about; template specialization explosion

---

## Conclusion

### **Recommendation: Keep separate for now, plan extraction for later**

**Immediate (Phases 1-8)**: Maintain ParallelFFT2D and ParallelFFT3D as independent specializations. Benefits:
- Clear, maintainable implementations
- Fast compilation
- Simple APIs
- Proven correct for deployed use cases

**Medium-term (4D or 3D slab needed)**: Extract non-template base class for common machinery (normalization, validation). This gives ~50% code reuse without template complexity.

**Long-term (if 4+ implementations exist)**: Unify to generic template with explicit specialization list. Only then does the template machinery pay off.

**Rationale**: Premature unification adds cognitive and compilation burden for questionable current benefit. The HPC domain values stability and performance over abstract generality. Wait until the pattern is proven and the benefit is clear.
