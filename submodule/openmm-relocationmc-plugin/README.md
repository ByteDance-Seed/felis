# OpenMM RelocationMC Plugin

`RelocationMCPlugin` defines a force-free OpenMM protocol for relocating complete
labile molecules between a ligand-centred spherical region and bulk solvent. A
labile molecule may be a water, ion, or multi-site fragment. A move changes the
accepted molecule's coordinates and consistently rotates its internal velocity
field; particle count and force-field parameters remain unchanged.

Current release: `0.1.0`. Supported release stack: OpenMM `>=8.5,<8.6`,
CUDA Toolkit `>=12.2`, and PerturbNonbondedPlugin `0.1.0`.

Unless a source file states otherwise, this plugin is licensed under the
Creative Commons Attribution-NonCommercial 4.0 International License
(`CC-BY-NC-4.0`); see [LICENSE](LICENSE).

## Protocols

New `RelocationMC` objects default to the `MTM` protocol.  `Legacy` remains an
explicit compatibility mode for reproducing historical trajectories: it uses
target-region anchor sampling, the prepared absolute orientation table, and
ordered first-success Bernoulli tests.  Its reverse source partition is a
pre-move approximation, and it is not claimed to satisfy batch detailed
balance.

The default `MTM` protocol implements paired-pool multiple-try Metropolis:

1. classify each real molecule and select a source molecule with
   `exp(+beta*s_D)/Z_S(x)`;
2. generate exactly `K=numCandidateSitesPerExchange` independent whole-box
   uniform anchors, each with an independent Haar-uniform rotation of the actual
   selected molecule's PBC-safe relative conformation;
3. classify, but never conditionally resample, every pose as inner or outer;
4. select one target-region pose by GPU Gumbel-Max with weight
   `exp(-beta*u_i)`, where `u_i` is its interaction with the common environment
   after removing `D`;
5. form the paired reverse pool by replacing the selected slot with the exact
   old pose and retaining every other slot, then compute `W_f` and `W_r`;
6. recompute every old target molecule's source score in the proposed state and
   include the selected molecule's `s_C=u_C+r(C)` to obtain exact `Z_T(y)`;
7. perform one final fail-closed acceptance test with

```
log r = log(g_r/g_f) + log Z_S(x) - log Z_T(y)
      + beta*(s_C-s_D) + log W_f - log W_r
      - beta*(DeltaU-(u_C-u_D)).
```

The common-environment part is `u_C-u_D`.  Controlled Context-energy oracles
show that the current finite-grid PME Hamiltonian also contains
`isolatedReciprocalDelta=0.5*(r(C)-r(D))`; MTM therefore includes that value as
`energyResidual` in the last term.  `r(pose)` is the isolated-molecule reciprocal
mesh contribution used by the historical full-grid source score, not the
coordinate-independent analytic Ewald self correction.  The analytic Ewald
self, net-charge correction, and rigid zero-exception terms cancel separately.
See
`docs/MTM_AUGMENTED_POOL_DERIVATION.md` for the augmented-state proof, including
the categorical selection ratio and cancellation of the common whole-box/Haar
proposal density.  Consequently, MTM has no explicit `V_T/V_S` factor.

Legacy still uses `K=candidateSites*60` when prepared orientations are enabled,
or `K=candidateSites` when disabled.  MTM always uses one independently sampled
Haar orientation per anchor and therefore interprets the configured candidate
site count directly as `K`.  Protocol, candidate count, and orientation setting
are Context-construction properties; changing them requires reinitialization.

### Accepted-molecule velocities

The production default rotates the accepted molecule's velocities with the same
rigid-body orientation change as its coordinates.  For site masses `m_i`, source
velocities `v_i`, and the proper orthogonal source-to-candidate rotation `Q`, the
committed velocities are

```
V_COM = sum_i(m_i*v_i)/sum_i(m_i)
v_i'  = V_COM + Q*(v_i-V_COM)
```

Thus the moved molecule's total linear momentum and mass-weighted kinetic energy
relative to its COM are preserved.  The COM velocity itself is not rotated or
resampled.  Rejected proposals leave every velocity unchanged, and an accepted
proposal changes only the selected molecule's atoms.  This supersedes the legacy
behavior that retained each moved atom's lab-frame velocity unchanged; there is
no zeroing or Maxwell-Boltzmann resampling compatibility mode.

`Q` is computed from minimum-image local vectors of the source molecule and the
actual accepted candidate coordinates downloaded from the CUDA proposal.  Those
coordinates are the definitive target frame, so the transform is independent of
candidate-index and template-origin conventions.  For 3-site water and other
nonlinear molecules, a proper orthogonal least-squares Horn fit over all local
vectors provides the closest rigid rotation when normal constraint/PDB/float
noise prevents an exact map, including when source or target coordinates straddle
a periodic boundary.  Linear
molecules use the minimum-angle rotation between their old and new axes (with a
deterministic perpendicular axis for an exactly antiparallel move), leaving the
unobservable rotation about the molecular axis unchanged.  Single-site or fully
coincident molecules use identity, which preserves their sole COM velocity.
Every registered labile molecule must have positive total mass.  A selected
source must also be sufficiently rigid-congruent to the configured reference:
all pair distances are checked before any Context mutation with
`1e-3 nm + 1e-4*referenceDistance` tolerance.  The `1e-3 nm` (`0.01 A`) absolute
term rejects gross distortions while remaining over five times the largest pair
residual observed in a 13,753-water production input scan.  A proper orthogonal
least-squares fit minimizes local-vector error.  Its residual is diagnostic, not
a second rejection criterion: pair-congruent coordinate sets with normal
constraint/PDB/float noise need not admit an exact rigid map.  A more strongly
distorted source raises an exception rather than silently assigning an ambiguous
rotation.
The implementation does not call `applyConstraints()` or otherwise alter the
accepted position or velocity proposal.

Velocity rotation is computed and validated before the accepted coordinates are
committed.  The current CUDA implementation uses OpenMM's full-system
`ContextImpl::getVelocities()` and `setVelocities()` interfaces, so each accepted
move adds one velocity download and upload/synchronization.  Rejected attempts
incur no velocity-transfer cost.
All transformation and congruence failures occur before `setPositions()`.  The
OpenMM position and velocity setters are separate calls, however, so a platform
failure specifically during `setVelocities()` can leave the already accepted
positions installed; this residual two-call non-atomicity is not hidden by an
untestable rollback path.

All labile molecules must have identical topology and site ordering. Site
parameters use charge in elementary-charge units, sigma in nm, and epsilon in
kJ/mol. `setLabileMolReferencePositions()` supplies positions relative to the
candidate anchor point. `ligandAtoms` are atoms of one ligand used to define one
sphere centre: the CUDA implementation unwraps them relative to the first atom,
takes their geometric mean, then wraps the unique centre back into the box.
RelocationMC currently supports orthorhombic periodic boxes only; Context
creation or parameter updates reject triclinic boxes.  PME environment-interaction mode also
requires every pair of atoms within a labile molecule to have a zeroed
`PerturbNonbondedForce` exception (`chargeProd == 0` and `epsilon == 0`); nonzero
intramolecular/1-4 nonbonded interactions are not supported.  Exceptions
between different labile molecules or between labile and fixed environment
atoms are also rejected in environment-interaction mode.  Fixed non-labile protein, ligand,
membrane, and solvent atoms with ordinary pair interactions remain part of the full-system
PME direct and reciprocal environment.  Historical RF-delete/uncorrected-PME
behavior is available only through the explicit
`setUseLegacyAsymmetricEnergies(true)` compatibility switch.

## Minimal API setup

```cpp
using namespace OpenMM;
using namespace OpenMM::RelocationMCPlugin;

RelocationMC sampler(300.0, 0.50, 100, 300);
sampler.setRandomSeed(20260820);
sampler.setLigandAtoms(ligandHeavyAtomIndices);
sampler.setLabileMolSiteParameters({
    {-0.834, 0.315075, 0.635968},
    { 0.417, 0.0,      0.0     },
    { 0.417, 0.0,      0.0     }
});
sampler.setLabileMolReferencePositions({
    Vec3(0.0, 0.0, 0.0),
    Vec3(0.0757, 0.0520, 0.0),
    Vec3(-0.0757, 0.0520, 0.0)
});
for (const auto& water : exchangeableWaters)
    sampler.addLabileMol({water.oxygen, water.hydrogen1, water.hydrogen2}, water.oxygen);
system.addForce(new RelocationMC(sampler));
```

The four-argument constructor defaults to `RelocationMC::MTM`, with Haar
orientations enabled, cavity bias disabled, and exact PME environment energies.
Legacy behavior must be selected explicitly before enabling its compatibility
settings:

```cpp
sampler.setProtocolVersion(RelocationMC::Legacy);
sampler.setUseCavityBias(true);
```

MTM requires orientations enabled and the environment-only exact-PME energy
path, so `setOrientationsEnabled(false)`, `setUseCavityBias(true)`, and
`setUseLegacyAsymmetricEnergies(true)` are rejected while MTM is selected.
It also requires ordinary, lambda-invariant labile particles whose A/B
charge, sigma, and epsilon exactly match the registered site parameters; dummy,
REST2-scaled, parameter-offset, cross-molecule exception, and disabled-direct-
space configurations fail during Context creation.  Constraints and bonded
terms may be wholly inside one moved rigid molecule or wholly outside all
labile molecules, but may not cross those boundaries.  Unknown additional
position-dependent Force implementations are rejected until their invariance or
energy contribution is implemented explicitly.

MTM distinguishes state geometry from the coordinates used by OpenMM's current
numerical Hamiltonian.  Single precision stores float4; mixed precision retains
both float4 base and float4 correction through actual-D extraction, Haar
rotation, paired reverse support, and accepted commit; double precision stores
double4.  Acceptance energies nevertheless use the same
`ContextEnergyCoordinates` as the active OpenMM force calculation: mixed uses
the quantized float4 base (and the same float/A&S direct formula), while double
uses double4.  The correction is therefore part of the represented state and
reverse round trip, but is not mixed into a base-only main PME grid.  This avoids
an invalid `full(base)-deleted(base+correction)` subtraction and defines detailed
balance relative to OpenMM's implemented numerical Hamiltonian.

Protocol changes require Context reinitialization.  Serialization schema v4
stores `protocolVersion`; schema v2/v3 files always deserialize as `Legacy`.

`exchangeFrequency == 0` disables automatic attempts. Manual attempts use
`attemptExchange(context)`. Automatic attempts begin only after the first
integration step, run at positive step multiples of the frequency, and execute
at most once at a given Context step.

With verbose logging enabled, every manual invocation or due automatic MC call
starts with the exact separator `-----N------`, where `N` is the Context-local log
record count.  This sequence advances even if no eligible proposal exists, while
the record keeps `attempted=0`; `proposalCount` advances only when
`ExchangeResult.attempted` is true.  Accepted records additionally contain
pre-commit and post-commit full OpenMM Context potential energies, plus the delete
and insert energies actually used by the selected legacy or environment-
interaction acceptance path. All energy keys carry the `_kJ_per_mol` suffix. The
existing pre-attempt full-Hamiltonian evaluation supplies the initial value
without additional logging overhead. Verbose mode adds one full Context evaluation
after each accepted coordinate commit; rejected and nonverbose attempts add no
post-commit evaluation.

## Python MC/MD example

`python/example_relocationmc_simulation.py` is an argparse-driven Simulation
example. It supports either a serialized PerturbNonbonded System + State with a
matching topology PDB, or a PDB + ForceField path that converts one standard
`NonbondedForce` into `PerturbNonbondedForce`. It registers homogeneous water or
other labile residues and refreshes the PME grid before MD. RelocationMC then
runs automatically through `updateContextState`; the loop does not call
`attemptExchange()` manually.

```bash
python python/example_relocationmc_simulation.py \
  --system-xml system.xml --state-xml state.xml --topology-pdb topology.pdb \
  --ligand-resname LIG --labile-resname HOH \
  --labile-atom-names O,H1,H2 --reference-atom-name O \
  --exchange-frequency 1000 --candidate-count 300 \
  --steps 10000 --report-interval 1000
```

Use `--ligand-center-indices` when residue-based centre selection is ambiguous.
The CUDA example leaves the default separate PME stream enabled.

## Building and installing

CUDA is required.  Install PerturbNonbondedPlugin first, then run the
RelocationMC source installer.  Before configuring a build, it verifies the
installed PerturbNonbonded public and internal headers, core library, CUDA
plugin, and native shared-library dependencies:

```bash
../perturb-nonbonded-force/install.sh
./install.sh
```

Use `./install.sh --help` for custom prefixes, build directories, Python
locations, and explicit CUDA-test control.  RelocationMC currently has only a
CUDA execution backend, so configuration and installation fail rather than
creating a nonfunctional CPU-only installation when CUDA is unavailable.

For a manual build against installed dependencies:

```bash
cmake -S . -B build \
  -DOPENMM_DIR=/path/to/openmm \
  -DPERTURB_NONBONDED_INSTALL_DIR=/path/to/perturb-prefix
cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake --install build
```

CUDA builds have single-, mixed-, and double-precision test coverage for the
PME environment-interaction removal/insertion path and both PME stream modes,
but the implementation is not a multi-device implementation.

`perturb_nonbonded_force` provides the zero-copy CUDA evaluator interfaces the
protocol requires: compatibility reaction-field queries, real/candidate
PME direct-space queries, isolated scratch-grid PME source maps and reciprocal
self energies, and full-grid PME potential interpolation. These interfaces are
currently limited to a single CUDA device and orthorhombic boxes. PME
interpolation requires GPU PME and a prior Context energy evaluation to populate
the grid. Both the default separate PME stream and `DisablePmeStream=true` are
supported; the query waits on the PME completion event before reading the grid.
CPU PME remains unsupported.

## Release status and limitations

The CUDA backend depends on OpenMM CUDA implementation headers and on the
experimental CUDA-query interface of the same PerturbNonbondedPlugin release.
Test every supported OpenMM/CUDA combination before publishing binaries.  The
current implementation supports one CUDA device, GPU PME, and orthorhombic
periodic boxes.  It does not provide Reference, CPU, OpenCL, CPU-PME, or
multi-device execution.
