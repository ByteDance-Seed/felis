# Paired-pool MTM derivation

This note records the augmented-state argument implemented by the MTM protocol.
It defines the algorithmic contract; it is not a substitute for independent
scientific review of a release candidate.

Let `x` be the current configuration, let `D` be the selected labile molecule,
and let `y` be the configuration obtained by replacing `D` with candidate pose
`C`.  The inner and outer regions form a binary partition.  The source molecule
is selected from its current region with normalized score

```text
p(D | x) = exp(beta*s_D)/Z_S(x).
```

The proposal draws an ordered pool of `K` independent poses from the same
whole-box translation and Haar-orientation density `h0`.  Poses are classified
after drawing and are never conditionally resampled.  Eligible target-region
poses have importance weights

```text
w_i = exp(-beta*u_i)/h0(i).
```

The implementation uses a uniform whole-box and Haar proposal, so `h0` is the
same constant for every pose and cancels from normalized selection and weight
ratios.  This is why no explicit inner/outer volume factor appears.

If slot `j` containing `C` is selected, the reverse pool is the same ordered
pool with slot `j` replaced by the exact old pose `D`.  This replacement is an
involution: applying the reverse construction restores the forward pool.  With
`W_f` and `W_r` denoting the eligible forward and reverse weight sums, the pool
density, categorical-selection probability, and target-density terms reduce to
the `W_f/W_r` contribution used by multiple-try Metropolis.  The finite-state
enumeration in `tests/TestMTMAugmentedPool.cpp` verifies this event pairing,
normalization, stationarity, and detailed balance for nonuniform `h0`, including
the `K=1` reduction to ordinary independence Metropolis-Hastings.

For molecular configurations, source selection and the final Context energy
change add the remaining factors.  The implemented log ratio is

```text
log r = log(g_r/g_f) + log Z_S(x) - log Z_T(y)
      + beta*(s_C-s_D) + log W_f - log W_r
      - beta*(DeltaU-(u_C-u_D)).
```

Here `g_f` and `g_r` are the probabilities of selecting the forward and reverse
region directions, `s_D` and `s_C` are full source scores, `u_D` and `u_C` are
interactions with the common environment, and `DeltaU` is the full OpenMM
potential-energy difference.

The reciprocal PME calculation is a quadratic form.  Writing the background as
`B`, its exact expansion gives

```text
U(B+C)-U(B+D) = u_C-u_D + 0.5*(r(C)-r(D)).
```

The second term is the isolated finite-grid reciprocal contribution, not the
coordinate-independent analytic Ewald self term.  The implementation records it
as `isolatedReciprocalDelta` and includes it in `energyResidual`.  The same test
file verifies the one-half coefficient with a finite symmetric reciprocal
operator.  CUDA Context-energy oracle tests then compare the decomposition with
OpenMM's numerical Hamiltonian in single, mixed, and double precision.

The acceptance probability is `min(1, exp(log r))`.  Empty eligible pools and
nonfinite weights reject without mutating the Context.  The proof relies on all
runtime restrictions checked at Context construction: homogeneous rigid labile
molecules, invariant registered nonbonded parameters, supported bonded-force
boundaries, zero unsupported exceptions, orthorhombic periodic geometry, and
the exact GPU-PME query path.
