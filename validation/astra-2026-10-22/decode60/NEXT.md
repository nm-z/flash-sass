# Current work toward60 TPS

Retained scalar default: v104,25.721831473 and25.792883107TPS,492 phases/token,
maximum pairwise assigned-work ratio1.076923077. Two short requests and the4K
fixture pass; reference outputs, state hashes, head audits and work audits are
recorded in README.md. Physical GPUs2 and7 remain reserved.60TPS is not achieved.

Nate has now explicitly authorized lossless MTP/speculative decoding with all
work included in throughput and phase accounting. The main work is the balanced
full-model verifier and lossless controller, not additional unchecked draft
rollouts. See SPECULATIVE.md for implemented code, unvalidated assumptions and
the concrete remaining integration/recovery work.

The balanced batch descriptor builder and nested production kernel frame compile.
They are not wired into a runnable engine mode yet. Do not remove the scalar-only
runtime guard without completing per-position buffer lifetime, acceptance,
rejection, state restoration, PLE restaging and honest work counters together.
No partial batch GPU run has been performed.

The router scheduling experiment is also complete. It preserves the lowest-index
tie rule and original routing-weight polynomial/sum order, with batched independent
comparisons and exponent chains. v105 measured25.597726149 and25.761084376TPS;
both requests match128 reference outputs and288 state hashes each. It did not
show a clear throughput gain and remains in `die_balanced_router.cubin`.

Objective weights remain throughput1.0, fidelity1.0, phase/work constraints1.0,
engineering time0.8 and cost0.8. Further scalar scheduling was the lower-effort
alternative; its latest router result did not materially close the gap. The
higher-effort speculative path can amortize target work, but only with complete
acceptance/recovery and measured acceptance. It is the selected next approach.
