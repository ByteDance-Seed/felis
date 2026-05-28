# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging

import openmm as mm
import openmm.unit as unit

logger = logging.getLogger(__name__)


class FIRE2Integrator(mm.CustomIntegrator):
    """Perform energy minimization using the FIRE 2.0 algorithm.

    The FIRE 2.0 method adapts the effective step size and velocity mixing
    based on the "power" P = sum(v * f), where v is velocity and f is force.
    The implementation uses OpenMM reduction operations (`addComputeSum`) to
    compute dot products and norms.

    This class is an OpenMM `CustomIntegrator`. It updates the OpenMM per-DOF
    variables `x` (positions) and `v` (velocities) in-place when stepping.

    Attributes:
        N_min (int): Minimum number of consecutive downhill steps (P > 0) before
            allowing time step growth.
        f_inc (float): Multiplicative factor used to increase `dt_var` after
            sufficient downhill steps.
        f_dec (float): Multiplicative factor used to decrease `dt_var` when
            uphill motion is detected (P <= 0).
        alpha_start (float): Initial velocity mixing parameter.
        f_alpha (float): Multiplicative decay factor for `alpha` after
            sufficient downhill steps.
        dt_max_val (float): Maximum allowed `dt_var` in picoseconds.
    """

    def __init__(
        self,
        dt_start=0.1 * unit.femtoseconds,
        dt_max=1.0 * unit.femtoseconds,
        *,
        # --- Default Constants from FIRE 2.0 Paper, DO NOT CHANGE ---
        tolerance=1e-5,
        N_min=20,
        f_inc=1.1,
        f_dec=0.5,
        alpha_start=0.1,
        f_alpha=0.99,
    ):
        """Initialize the FIRE 2.0 minimizer integrator.

        Args:
            dt_start (openmm.unit.Quantity): Initial integration time step.
                Units: time (e.g., femtoseconds). Stored internally as
                picoseconds in the global variable `dt_var`.
            dt_max (openmm.unit.Quantity): Maximum allowed adaptive time step.
                Units: time (e.g., femtoseconds). Converted to picoseconds and
                used as an upper bound for `dt_var`.
            tolerance (float): Constraint tolerance passed to
                `CustomIntegrator.setConstraintTolerance()`. This is
                dimensionless.
            N_min (int): Minimum number of consecutive downhill steps (P > 0)
                before increasing the adaptive time step.
            f_inc (float): Multiplicative factor used to increase the adaptive
                time step after sufficient downhill steps.
            f_dec (float): Multiplicative factor used to decrease the adaptive
                time step when uphill motion is detected (P <= 0).
            alpha_start (float): Initial mixing parameter `alpha` used in the
                FIRE velocity mixing step.
            f_alpha (float): Multiplicative decay factor applied to `alpha`
                after sufficient downhill steps.

        Raises:
            AttributeError: If the imported `openmm` module does not provide
                `CustomIntegrator`.
        """
        super(FIRE2Integrator, self).__init__(dt_start)
        # Critical: Set constraint tolerance for rigid bonds
        self.setConstraintTolerance(tolerance)
        self.N_min = N_min
        self.f_inc = f_inc
        self.f_dec = f_dec
        self.alpha_start = alpha_start
        self.f_alpha = f_alpha
        self.dt_max_val = dt_max.value_in_unit(unit.picoseconds)

        logger.info(
            "FIRE2Integrator config: dt_start=%s  dt_max=%s  "
            "constraint_tolerance=%s (dimensionless) N_min=%s f_inc=%s f_dec=%s alpha_start=%s f_alpha=%s "
            "(integrator internal time unit: ps)",
            dt_start,
            dt_max,
            tolerance,
            N_min,
            f_inc,
            f_dec,
            alpha_start,
            f_alpha,
        )
        # --- 1. Define Global State Variables ---
        self.addGlobalVariable("dt_var", dt_start.value_in_unit(unit.picoseconds))
        self.addGlobalVariable("alpha", self.alpha_start)
        self.addGlobalVariable("n_pos", 0)  # Steps since P > 0
        # Variables for reduction results
        self.addGlobalVariable("P", 0.0)  # Power
        self.addGlobalVariable("fnorm", 0.0)  # Force Norm
        self.addGlobalVariable("vnorm", 0.0)  # Velocity Norm
        self.addGlobalVariable("sum_f2", 0.0)  # Sum(f*f)
        self.addGlobalVariable("sum_v2", 0.0)  # Sum(v*v)
        # --- 2. Compute Norms and Power (The Fix) ---
        # 'addComputeSum' sums the expression over all degrees of freedom
        # and stores it in the global variable.
        self.addComputeSum("sum_f2", "f*f")
        self.addComputeSum("sum_v2", "v*v")
        self.addComputeSum("P", "v*f")
        # Now compute the derived global values using standard math
        self.addComputeGlobal("fnorm", "sqrt(sum_f2)")
        self.addComputeGlobal("vnorm", "sqrt(sum_v2)")
        # --- 3. Logic: Uphill Motion (P <= 0) ---
        self.beginIfBlock("P <= 0")
        self.addComputeGlobal("dt_var", f"dt_var * {self.f_dec}")
        self.addComputeGlobal("alpha", f"{self.alpha_start}")
        self.addComputeGlobal("n_pos", "0")
        self.addComputePerDof("v", "0")  # Freeze velocities
        self.endBlock()
        # --- 4. Logic: Downhill Motion (P > 0) ---
        self.beginIfBlock("P > 0")
        self.addComputeGlobal("n_pos", "n_pos + 1")
        # Only accelerate if stable for N_min steps
        self.beginIfBlock(f"n_pos > {self.N_min}")
        self.addComputeGlobal("dt_var", f"min(dt_var * {self.f_inc}, {self.dt_max_val})")
        self.addComputeGlobal("alpha", f"alpha * {self.f_alpha}")
        self.endBlock()
        self.endBlock()
        # --- 5. Velocity Mixing (FIRE Core) ---
        # v = (1-alpha)*v + alpha * (F / |F|) * |v|
        # Safety: (fnorm + 1e-10) avoids division by zero
        self.addComputePerDof("v", "v*(1-alpha) + (f/(fnorm+1e-10))*vnorm*alpha")
        # --- 6. Integration (Euler Semi-Implicit) ---
        # A. Update Velocity
        self.addComputePerDof("v", "v + (f/m)*dt_var")
        self.addConstrainVelocities()
        # B. Update Position using NEW velocity
        self.addComputePerDof("x", "x + v*dt_var")
        # --- 7. Final Constraint Projection ---
        self.addConstrainPositions()
        self.addConstrainVelocities()
