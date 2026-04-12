// cart_sub_test.cpp - Phase 5C validation: Cartesian sub-communicator
// correctness on a 2x3 grid (6 ranks).
//
// Layout (dims = {2, 3}, row-major coordinates):
//
//   world rank | (axis 0, axis 1) | meaning
//   -----------+------------------+----------
//      0       |     (0, 0)       | row 0, col 0
//      1       |     (0, 1)       | row 0, col 1
//      2       |     (0, 2)       | row 0, col 2
//      3       |     (1, 0)       | row 1, col 0
//      4       |     (1, 1)       | row 1, col 1
//      5       |     (1, 2)       | row 1, col 2
//
// What this test validates:
//
//   Phase 1 - cart_create assigns the expected coordinates to every rank.
//
//   Phase 2 - cart_sub(1) ("vary axis 1, share axis 0") = ROW sub-comms.
//             Two cohorts (one per row), each of size 3. Each row's
//             sub-rank 0 (col 0) broadcasts a row-specific payload.
//             Receivers verify they got their row's value, not the other.
//
//   Phase 3 - cart_sub(0) ("vary axis 0, share axis 1") = COL sub-comms.
//             Three cohorts (one per column), each of size 2. Each column's
//             sub-rank 0 (row 0) broadcasts a col-specific payload.
//
//   Phase 4 - ISOLATION: row.bcast and col.bcast back-to-back at the SAME
//             default tag. Both senders are sub-rank 0 in their respective
//             sub-comms with src=0 in the wire header. The only thing keeping
//             these two streams apart is the comm_id propagated through the
//             central dispatch loop's per-key channel lookup. If that
//             demultiplexing is wrong, this phase will fail.
//
//   Phase 5 - Sub-comm barriers: each row and each column barriers
//             independently of the other and of the world.
//
// Requires: 6 ranks, CLUSTR_RECV=CLUSTR_RECV_CENTRAL (cart_sub static-asserts
// under the inline path). Compiles under both CLUSTR_TRANSPORT modes.
//
// On a small dev cluster, run via tests/run_cart_sub_local.sh which spawns
// 6 localhost processes wired through 6 generated rosters. Once you have a
// 6-machine cluster you can submit it normally with Ranks=6.

#include "clustr_mpi.h"
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

// Print + abort. Coroutine return-from-deep-nesting plus the world barrier
// pattern means a single failing rank would otherwise hang the others on the
// next barrier; immediate process exit closes our sockets so peers see EPIPE
// on their next op and abort fast.
[[noreturn]] void die(int rank, int code, const std::string& msg) {
    std::cerr << "[rank " << rank << "] FAIL: " << msg << "\n" << std::flush;
    std::_Exit(code);
}

}  // namespace

CLUSTR_MPI_MAIN(mpi) {
    const int my = mpi.rank();
    const int n  = mpi.size();

    if (n != 6) {
        if (my == 0)
            std::cerr << "[cart_sub_test] FATAL: requires exactly 6 ranks (got "
                      << n << ")\n";
        co_return 1;
    }

    auto& world = mpi.world();

    // ── Phase 1: cart_create ────────────────────────────────────────────
    world.cart_create({2, 3});

    const int expected_row = my / 3;  // axis 0
    const int expected_col = my % 3;  // axis 1
    if (world.coords().size() != 2 ||
        world.coords()[0] != expected_row ||
        world.coords()[1] != expected_col) {
        die(my, 2, "cart_create coords mismatch");
    }

    co_await world.barrier();
    if (my == 0) std::cout << "[cart_sub_test] phase 1 cart_create OK\n";

    // ── Phase 2: row sub-comm bcast ─────────────────────────────────────
    auto row = world.cart_sub(1);
    if (row.size() != 3 || row.rank() != expected_col)
        die(my, 3, "row sub-comm shape wrong");

    {
        // Each row's sub-rank-0 (world rank 3 * expected_row) broadcasts a
        // payload that encodes its world rank, so receivers can prove they
        // got their own row's frame and not the neighbor row's.
        const int row_root_world = expected_row * 3;
        std::vector<int> buf;
        if (row.rank() == 0) {
            buf = { 1000 + row_root_world,
                    1000 + row_root_world + 1,
                    1000 + row_root_world + 2 };
        }
        co_await row.bcast(buf, 0);

        const std::vector<int> expected = {
            1000 + row_root_world,
            1000 + row_root_world + 1,
            1000 + row_root_world + 2,
        };
        if (buf != expected)
            die(my, 4, "row.bcast payload mismatch");
    }

    co_await world.barrier();
    if (my == 0) std::cout << "[cart_sub_test] phase 2 row.bcast OK\n";

    // ── Phase 3: col sub-comm bcast ─────────────────────────────────────
    auto col = world.cart_sub(0);
    if (col.size() != 2 || col.rank() != expected_row)
        die(my, 5, "col sub-comm shape wrong");

    {
        const int col_root_world = expected_col;  // row 0 of my col
        std::vector<int> buf;
        if (col.rank() == 0) {
            buf = { 2000 + col_root_world,
                    2000 + col_root_world + 10 };
        }
        co_await col.bcast(buf, 0);

        const std::vector<int> expected = {
            2000 + col_root_world,
            2000 + col_root_world + 10,
        };
        if (buf != expected)
            die(my, 6, "col.bcast payload mismatch");
    }

    co_await world.barrier();
    if (my == 0) std::cout << "[cart_sub_test] phase 3 col.bcast OK\n";

    // ── Phase 4: ISOLATION (row + col bcast, same tag) ──────────────────
    //
    // Both row.bcast and col.bcast issue a SEND from sub-rank-0 with the
    // same default bcast tag (-2). On the wire both frames carry src=0,
    // tag=-2, type=SEND — only the comm_id distinguishes them. If recv
    // demultiplexing ignored comm_id (or used the wrong central state),
    // a row receiver could pull a col frame off the channel and blow up
    // size mismatches don't even guarantee a crash. So we sanity-check the
    // exact bytes.
    {
        const int row_root_world = expected_row * 3;
        const int col_root_world = expected_col;

        std::vector<int> rb, cb;
        if (row.rank() == 0) rb = { 30000 + row_root_world };
        if (col.rank() == 0) cb = { 40000 + col_root_world };

        co_await row.bcast(rb);   // default tag (-2)
        co_await col.bcast(cb);   // default tag (-2)

        if (rb.size() != 1 || rb[0] != 30000 + row_root_world)
            die(my, 7, "phase 4 row payload corrupted by col stream");
        if (cb.size() != 1 || cb[0] != 40000 + col_root_world)
            die(my, 8, "phase 4 col payload corrupted by row stream");
    }

    co_await world.barrier();
    if (my == 0) std::cout << "[cart_sub_test] phase 4 isolation OK\n";

    // ── Phase 5: independent sub-comm barriers ──────────────────────────
    //
    // Each row and each column hits its own barrier. Under CENTRAL, the
    // BARRIER tokens flow through the same shared sockets as everything
    // else and the comm_id+rank+tag tuple keeps each cohort's tokens in
    // its own per-key channel. A bug here would manifest as a hang.
    co_await row.barrier();
    co_await col.barrier();
    co_await world.barrier();
    if (my == 0) std::cout << "[cart_sub_test] phase 5 sub-comm barriers OK\n";

    if (my == 0) std::cout << "[cart_sub_test] ALL PHASES PASSED\n";
    co_return 0;
}
