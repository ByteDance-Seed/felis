// Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifdef CONTEXT_PRECISION_POSITIONS
#define SCRATCH_POSITION_BASE_TYPE real4
#define SCRATCH_REAL mixed
#define SCRATCH_REAL3 mixed3
#define MAKE_SCRATCH_REAL3 make_mixed3
#else
#define SCRATCH_POSITION_BASE_TYPE float4
#define SCRATCH_REAL real
#define SCRATCH_REAL3 real3
#define MAKE_SCRATCH_REAL3 make_real3
#endif

inline __device__ SCRATCH_REAL3 loadScratchPosition(
        GLOBAL const SCRATCH_POSITION_BASE_TYPE* RESTRICT positionBase,
#ifdef CONTEXT_PRECISION_POSITIONS
        GLOBAL const float4* RESTRICT positionCorrection,
#endif
        int index) {
    const SCRATCH_POSITION_BASE_TYPE base = positionBase[index];
#ifdef USE_MIXED_PRECISION
#ifdef CONTEXT_PRECISION_POSITIONS
    const float4 correction = positionCorrection[index];
    return MAKE_SCRATCH_REAL3((SCRATCH_REAL) base.x+(SCRATCH_REAL) correction.x,
            (SCRATCH_REAL) base.y+(SCRATCH_REAL) correction.y,
            (SCRATCH_REAL) base.z+(SCRATCH_REAL) correction.z);
#endif
#endif
    return MAKE_SCRATCH_REAL3((SCRATCH_REAL) base.x, (SCRATCH_REAL) base.y,
            (SCRATCH_REAL) base.z);
}

KERNEL void spreadScratchPmeCharges(
        int numSources,
        GLOBAL const SCRATCH_POSITION_BASE_TYPE* RESTRICT positions,
#ifdef CONTEXT_PRECISION_POSITIONS
        GLOBAL const float4* RESTRICT positionCorrection,
#endif
        GLOBAL const float* RESTRICT charges,
        GLOBAL real* RESTRICT grid,
        real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    for (int source = GLOBAL_ID; source < numSources; source += GLOBAL_SIZE) {
        SCRATCH_REAL3 position = loadScratchPosition(positions,
#ifdef CONTEXT_PRECISION_POSITIONS
                positionCorrection,
#endif
                source);
        APPLY_PERIODIC_TO_POS(position)
        SCRATCH_REAL3 t = MAKE_SCRATCH_REAL3(position.x*recipBoxVecX.x+position.y*recipBoxVecY.x+position.z*recipBoxVecZ.x,
                                             position.y*recipBoxVecY.y+position.z*recipBoxVecZ.y,
                                             position.z*recipBoxVecZ.z);
        t.x = (t.x-floor(t.x))*GRID_SIZE_X;
        t.y = (t.y-floor(t.y))*GRID_SIZE_Y;
        t.z = (t.z-floor(t.z))*GRID_SIZE_Z;
        int3 gridIndex = make_int3(((int) t.x)%GRID_SIZE_X, ((int) t.y)%GRID_SIZE_Y, ((int) t.z)%GRID_SIZE_Z);
        SCRATCH_REAL3 dr = MAKE_SCRATCH_REAL3(t.x-(int)t.x, t.y-(int)t.y, t.z-(int)t.z);
        SCRATCH_REAL3 data[PME_ORDER];
        const SCRATCH_REAL scale = RECIP((SCRATCH_REAL)(PME_ORDER-1));
        data[PME_ORDER-1] = MAKE_SCRATCH_REAL3(0);
        data[1] = dr;
        data[0] = MAKE_SCRATCH_REAL3(1)-dr;
        for (int order = 3; order < PME_ORDER; order++) {
            const SCRATCH_REAL div = RECIP((SCRATCH_REAL)(order-1));
            data[order-1] = div*dr*data[order-2];
            for (int k = 1; k < order-1; k++)
                data[order-k-1] = div*((dr+MAKE_SCRATCH_REAL3(k))*data[order-k-2]+(MAKE_SCRATCH_REAL3(order-k)-dr)*data[order-k-1]);
            data[0] = div*(MAKE_SCRATCH_REAL3(1)-dr)*data[0];
        }
        data[PME_ORDER-1] = scale*dr*data[PME_ORDER-2];
        for (int j = 1; j < PME_ORDER-1; j++)
            data[PME_ORDER-j-1] = scale*((dr+MAKE_SCRATCH_REAL3(j))*data[PME_ORDER-j-2]+(MAKE_SCRATCH_REAL3(PME_ORDER-j)-dr)*data[PME_ORDER-j-1]);
        data[0] = scale*(MAKE_SCRATCH_REAL3(1)-dr)*data[0];
        const SCRATCH_REAL charge = SQRT_ONE_4PI_EPS0*charges[source];
        for (int ix = 0; ix < PME_ORDER; ix++) {
            int xbase = gridIndex.x+ix;
            xbase -= (xbase >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            xbase *= GRID_SIZE_Y*GRID_SIZE_Z;
            for (int iy = 0; iy < PME_ORDER; iy++) {
                int ybase = gridIndex.y+iy;
                ybase -= (ybase >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                ybase = xbase+ybase*GRID_SIZE_Z;
                for (int iz = 0; iz < PME_ORDER; iz++) {
                    int zindex = gridIndex.z+iz;
                    zindex -= (zindex >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
                    atomicAdd(&grid[ybase+zindex], (real) (charge*data[ix].x*data[iy].y*data[iz].z));
                }
            }
        }
    }
}

KERNEL void reduceScratchCandidateSelfEnergy(
        int numAtoms,
        GLOBAL const float* RESTRICT charges,
        GLOBAL const float* RESTRICT potentials,
        GLOBAL float* RESTRICT output) {
    if (GLOBAL_ID != 0)
        return;
    float energy = 0.0f;
    for (int atom = 0; atom < numAtoms; atom++)
        energy += 0.5f*SQRT_ONE_4PI_EPS0*charges[atom]*potentials[atom];
    output[0] = energy;
}
