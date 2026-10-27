# Joint training followed by head-only updates

`--reverse_head_only_fraction .3` trains the backbone adapters and head jointly before freezing the adapters for the final 30% of updates. The default value of zero uses joint training throughout. This implementation validates the fixed recipe and rejects incompatible warm starts, heads, adapter scopes, auxiliary losses, and partial training horizons.

The complete horizon is 1,932 updates for F2 and 1,931 for F3. Joint training lasts `T - ceil(Fraction(str(.3)) * T)` updates: 1,352 for F2 and 1,351 for F3. After the boundary update and scheduler step, adapter parameters receive `requires_grad=False` and `grad=None`; the backbone switches to evaluation mode while the head remains trainable. Optimizer groups and state are retained, and the head continues the original OneCycle schedule.

Update records distinguish optimizer-group membership, trainable parameter count, and parameters with nonempty gradients. `reverse_phase_for_this_update` identifies the phase used for that update. Final phase counts must match the planned horizon before checkpoint saving. Initial and final tensor snapshots include the adapters.

`tests/test_reverse_schedule.py` checks the disabled schedule against joint training, the phase boundary, frozen adapter weights and optimizer moments, continuing head updates, and scheduler and RNG state. It also checks that the head-only phase clears stale adapter gradients and restores the required module modes before each forward pass.
