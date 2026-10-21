# GPUQ issue 106 visibility check

This Rust integration test starts one job in an alternate history database and
queues a second job in the default history database. Both use the same temporary
FIFO directory and lock path. It verifies that the FIFO is shared while each
`gpuq ls` result remains scoped to the database selected with `--db`.

The test runs the checked-in Ruby CLI but does not change or add Ruby code. Its
queue, lock, databases, and child processes are isolated under a temporary
directory; it does not access the host GPU queue.

Run the check on benji from the repository root:

```text
dsh build --host benji --require-tool cargo --require-tool ruby -- cargo test --manifest-path validation/issue106/Cargo.toml
```
