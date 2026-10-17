OpenMM PerturbNonbondedForce Plugin
===================================

This directory builds an OpenMM plugin that provides a custom Force class:

- `PerturbNonbondedPlugin::PerturbNonbondedForce`

Current release: `0.1.0`. Supported release stack: OpenMM `>=8.5,<8.6` and
CUDA Toolkit `>=12.2`.

It is designed for alchemical and REST2-style workflows where each particle can have *two* nonbonded parameter sets (A and B) and the effective interaction is controlled at runtime by Context parameters ("lambdas").


What the plugin does
====================

`PerturbNonbondedForce` computes Coulomb and Lennard-Jones (LJ) interactions similar to OpenMM's `NonbondedForce`, but with two per-particle parameter sets plus two boolean flags:

- Per particle: `(chargeA, sigmaA, epsilonA, chargeB, sigmaB, epsilonB, isDummy, isREST2)`.
- Long-range methods: `NoCutoff`, `CutoffNonPeriodic`, `CutoffPeriodic`, `Ewald`, `PME`.

At runtime the kernel reads these Context parameters:

- `lambdaA_charge`, `lambdaB_charge`, `lambdaC_charge`
- `lambdaA_vdw`, `lambdaB_vdw`
- `rest2_scale`

and uses them to compute the effective interactions.


Runtime parameterization (lambdas)
==================================

**Charges**

For each particle, the effective charge is computed as

```text
q = lambdaA_charge*qA + lambdaB_charge*qB + lambdaC_charge*qC
```

where `qC` is defined internally as:

- `qC = 0` if `isDummy == true`
- `qC = 0.5*(qA+qB)` if `isDummy == false`

**REST2 scaling**

If `isREST2 == true`, the force applies REST2-like scaling using `rest2_scale`:

- charges are scaled by `sqrt(rest2_scale)`
- LJ epsilons are scaled by `rest2_scale` (implemented as scaling `sqrt(epsilon)` by `sqrt(rest2_scale)`)

**Lennard-Jones (softcore for dummy atoms)**

If either particle in a pair has `isDummy == true`, a softcore-style LJ is used with `softcore_alpha` and the LJ contributions are scaled by `lambdaA_vdw` and `lambdaB_vdw`.

If neither particle is a dummy, the Reference implementation uses the A-state LJ parameters (no lambda scaling).


Force-group splitting
=====================

This force supports splitting direct and reciprocal space into different force groups for multiple time stepping:

- `setReciprocalSpaceForceGroup(int group)` controls where Ewald/PME reciprocal space is accumulated.
- `setIncludeDirectSpace(false)` disables direct-space terms (keeping reciprocal-only for PME/Ewald).

On CUDA, LJ and Coulomb direct-space terms can also be assigned separate force groups:

- `setDirectSpaceForceGroup(int group)` (Coulomb direct)
- `setLJForceGroup(int group)` (LJ direct)


Minimal C++ usage
=================

```cpp
#include "PerturbNonbondedForce.h"

using namespace OpenMM;
using namespace PerturbNonbondedPlugin;

auto* nb = new PerturbNonbondedForce();

// Required Context parameters (names must match exactly).
nb->addGlobalParameter("lambdaA_charge", 1.0);
nb->addGlobalParameter("lambdaB_charge", 0.0);
nb->addGlobalParameter("lambdaC_charge", 0.0);
nb->addGlobalParameter("rest2_scale",   1.0);
nb->addGlobalParameter("lambdaA_vdw",   1.0);
nb->addGlobalParameter("lambdaB_vdw",   0.0);

// Per particle: A and B parameters + flags.
nb->addParticle(chargeA, sigmaA, epsilonA, chargeB, sigmaB, epsilonB,
                /*isDummy=*/false, /*isREST2=*/false);

// Typical nonbonded settings.
nb->setNonbondedMethod(PerturbNonbondedForce::PME);
nb->setCutoffDistance(1.0);
nb->setSoftcoreAlpha(1.0);
```

At runtime you can change the parameters via:

```cpp
context.setParameter("lambdaB_charge", 0.5);
context.setParameter("lambdaB_vdw",   0.5);
context.setParameter("rest2_scale",   0.8);
```


Building and installing
=======================

CUDA is a required dependency.  The source installer verifies `nvcc` and the
OpenMM CUDA plugin, builds both Reference and CUDA implementations, runs the
available tests, and installs the native libraries and Python wrapper:

```bash
./install.sh
```

Use `./install.sh --help` to select prefixes, Python, build directories, and
whether CUDA runtime tests must run.  CUDA compilation is mandatory, while
`--cuda-tests auto` still permits compilation in a driverless build image.

For a manual native build:

```bash
cmake -S . -B build \
  -DOPENMM_DIR=/path/to/openmm \
  -DCMAKE_INSTALL_PREFIX=/path/to/openmm
cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake --install build
```

Notes:

- `OPENMM_DIR` must point to the OpenMM installation prefix so CMake can find headers and libraries.
- CMake configuration fails if CUDAToolkit or OpenMMCUDA is unavailable.
- The platform plugin libraries are installed into `${CMAKE_INSTALL_PREFIX}/lib/plugins`.
- The main library and headers are installed into `${CMAKE_INSTALL_PREFIX}/lib` and `${CMAKE_INSTALL_PREFIX}/include`.


Running tests
=============

```bash
ctest --test-dir build --output-on-failure
```

When linking directly (e.g., in standalone C++ tests), you may need to register kernels explicitly:

- `registerPerturbNonbondedReferenceKernelFactories()`
- `registerPerturbNonbondedCudaKernelFactories()`


Limitations / caveats
=====================

- LJ switching is not supported: `PerturbNonbondedForce::setUseSwitchingFunction(true)` throws.
- Particle/exception parameter offsets are not implemented (the corresponding API methods throw).
- Non-zero exception `chargeProd` or `epsilon` is not supported (use `CustomBondForce` for 1-4 terms if needed).
- `LJPME` is not supported (CUDA and Reference platforms).
- CUDA query APIs are experimental, require one CUDA device and GPU PME, and
  may need adaptation for new OpenMM releases because they cooperate with
  OpenMM's internal CUDA implementation.
- OpenCL is not implemented.


Authors
=======
- Ye Ding


## License

This plugin is licensed under the GNU Lesser General Public License version 3
or later (`LGPL-3.0-or-later`). See `LICENSE`, `COPYING`, and
`COPYING.LESSER`. Much of the plugin is adapted from the OpenMM 8.5.0
`NonbondedForce` source code; those portions retain the MIT or LGPL license
notices embedded in their source files.
