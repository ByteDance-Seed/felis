// Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: CC-BY-NC-4.0

#ifndef OPENMM_RELOCATIONMCVELOCITY_H_
#define OPENMM_RELOCATIONMCVELOCITY_H_

#include "openmm/OpenMMException.h"
#include "openmm/Vec3.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace OpenMM {
namespace RelocationMCPlugin {
namespace RelocationMCVelocity {

using RotationMatrix = std::array<Vec3, 3>;

// Reject grossly distorted source geometries, not ordinary constraint/PDB/float
// noise.  The 1e-3 nm (0.01 A) absolute tolerance is over five times the maximum
// 1.765e-4 nm pair residual observed across 13,753 production waters, yet still
// rejects the 0.002 nm distortion regression.  The relative term scales the same
// support check for larger rigid fragments.
constexpr double RIGID_ABSOLUTE_TOLERANCE_NM = 1.0e-3;
constexpr double RIGID_RELATIVE_TOLERANCE = 1.0e-4;

inline double rigidTolerance(double referenceDistance) {
    return RIGID_ABSOLUTE_TOLERANCE_NM+RIGID_RELATIVE_TOLERANCE*referenceDistance;
}

inline RotationMatrix identityRotation() {
    return {Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0)};
}

/** Apply a rotation stored as the images of the three Cartesian basis vectors. */
inline Vec3 applyRotation(const RotationMatrix& rotation, const Vec3& vector) {
    return rotation[0]*vector[0]+rotation[1]*vector[1]+rotation[2]*vector[2];
}

inline Vec3 minimumImage(const Vec3& vector, const Vec3 box[3]) {
    Vec3 result = vector;
    for (int axis = 0; axis < 3; axis++)
        result[axis] -= std::floor(result[axis]/box[axis][axis]+0.5)*box[axis][axis];
    return result;
}

inline Vec3 normalized(const Vec3& vector, const char* errorMessage) {
    const double norm = std::sqrt(vector.dot(vector));
    if (!(norm > 1.0e-12))
        throw OpenMMException(errorMessage);
    return vector*(1.0/norm);
}

/**
 * Return the minimum-angle proper rotation that maps one unit vector to another.
 * The antiparallel case uses a deterministic Cartesian perpendicular axis.
 */
inline RotationMatrix rotationBetweenUnitVectors(const Vec3& from, const Vec3& to) {
    double cosine = from.dot(to);
    if (cosine > 1.0)
        cosine = 1.0;
    else if (cosine < -1.0)
        cosine = -1.0;
    if (cosine > 1.0-1.0e-12)
        return identityRotation();

    auto rotate = [&](const Vec3& vector) {
        if (cosine < -1.0+1.0e-12) {
            int leastAlignedAxis = 0;
            if (std::fabs(from[1]) < std::fabs(from[leastAlignedAxis]))
                leastAlignedAxis = 1;
            if (std::fabs(from[2]) < std::fabs(from[leastAlignedAxis]))
                leastAlignedAxis = 2;
            Vec3 basis(0.0, 0.0, 0.0);
            basis[leastAlignedAxis] = 1.0;
            const Vec3 axis = normalized(from.cross(basis),
                    "RelocationMC could not define the antiparallel linear-molecule rotation axis.");
            return axis*(2.0*axis.dot(vector))-vector;
        }
        const Vec3 cross = from.cross(to);
        return vector*cosine+cross.cross(vector)+cross*(cross.dot(vector)/(1.0+cosine));
    };
    return {rotate(Vec3(1.0, 0.0, 0.0)), rotate(Vec3(0.0, 1.0, 0.0)),
            rotate(Vec3(0.0, 0.0, 1.0))};
}

/** Horn quaternion solution of the proper orthogonal least-squares fit. */
inline RotationMatrix bestFitProperRotation(const std::vector<Vec3>& sourceVectors,
        const std::vector<Vec3>& targetVectors) {
    double covariance[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    for (int site = 0; site < static_cast<int>(sourceVectors.size()); site++)
        for (int row = 0; row < 3; row++)
            for (int column = 0; column < 3; column++)
                covariance[row][column] += targetVectors[site][row]*sourceVectors[site][column];
    const double trace = covariance[0][0]+covariance[1][1]+covariance[2][2];
    double horn[4][4] = {
        {trace, covariance[1][2]-covariance[2][1], covariance[2][0]-covariance[0][2], covariance[0][1]-covariance[1][0]},
        {covariance[1][2]-covariance[2][1], covariance[0][0]-covariance[1][1]-covariance[2][2], covariance[0][1]+covariance[1][0], covariance[0][2]+covariance[2][0]},
        {covariance[2][0]-covariance[0][2], covariance[0][1]+covariance[1][0], -covariance[0][0]+covariance[1][1]-covariance[2][2], covariance[1][2]+covariance[2][1]},
        {covariance[0][1]-covariance[1][0], covariance[0][2]+covariance[2][0], covariance[1][2]+covariance[2][1], -covariance[0][0]-covariance[1][1]+covariance[2][2]}
    };
    // Symmetric 4x4 Jacobi eigensolve, selecting the largest algebraic
    // eigenvalue required by Horn's method (not the largest magnitude one).
    double eigenvectors[4][4] = {{1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0},
                                  {0.0, 0.0, 1.0, 0.0}, {0.0, 0.0, 0.0, 1.0}};
    for (int iteration = 0; iteration < 64; iteration++) {
        int first = 0, second = 1;
        double maximum = std::fabs(horn[first][second]);
        for (int row = 0; row < 4; row++)
            for (int column = row+1; column < 4; column++)
                if (std::fabs(horn[row][column]) > maximum) {
                    maximum = std::fabs(horn[row][column]);
                    first = row;
                    second = column;
                }
        if (maximum < 1.0e-15)
            break;
        const double angle = 0.5*std::atan2(2.0*horn[first][second],
                horn[second][second]-horn[first][first]);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        for (int row = 0; row < 4; row++) {
            if (row == first || row == second)
                continue;
            const double oldFirst = horn[row][first];
            const double oldSecond = horn[row][second];
            horn[row][first] = horn[first][row] = cosine*oldFirst-sine*oldSecond;
            horn[row][second] = horn[second][row] = sine*oldFirst+cosine*oldSecond;
        }
        const double oldFirstDiagonal = horn[first][first];
        const double oldSecondDiagonal = horn[second][second];
        const double oldOffDiagonal = horn[first][second];
        horn[first][first] = cosine*cosine*oldFirstDiagonal-2.0*sine*cosine*oldOffDiagonal+
                sine*sine*oldSecondDiagonal;
        horn[second][second] = sine*sine*oldFirstDiagonal+2.0*sine*cosine*oldOffDiagonal+
                cosine*cosine*oldSecondDiagonal;
        horn[first][second] = horn[second][first] = 0.0;
        for (int row = 0; row < 4; row++) {
            const double oldFirst = eigenvectors[row][first];
            const double oldSecond = eigenvectors[row][second];
            eigenvectors[row][first] = cosine*oldFirst-sine*oldSecond;
            eigenvectors[row][second] = sine*oldFirst+cosine*oldSecond;
        }
    }
    int largest = 0;
    for (int index = 1; index < 4; index++)
        if (horn[index][index] > horn[largest][largest])
            largest = index;
    double quaternion[4];
    for (int component = 0; component < 4; component++)
        quaternion[component] = eigenvectors[component][largest];
    const double w = quaternion[0], x = quaternion[1], y = quaternion[2], z = quaternion[3];
    return {
        Vec3(1-2*y*y-2*z*z, 2*x*y+2*w*z, 2*x*z-2*w*y),
        Vec3(2*x*y-2*w*z, 1-2*x*x-2*z*z, 2*y*z+2*w*x),
        Vec3(2*x*z+2*w*y, 2*y*z-2*w*x, 1-2*x*x-2*y*y)
    };
}

inline void validateRigidCongruence(const std::vector<Vec3>& referencePositions,
        const std::vector<Vec3>& sourcePositions, const Vec3 box[3]) {
    if (referencePositions.size() != sourcePositions.size())
        throw OpenMMException("RelocationMC source and reference conformations have different sizes.");
    double maximumResidual = 0.0;
    double toleranceAtMaximum = 0.0;
    for (int first = 0; first < static_cast<int>(referencePositions.size()); first++)
        for (int second = first+1; second < static_cast<int>(referencePositions.size()); second++) {
            const Vec3 referenceDelta = minimumImage(referencePositions[second]-referencePositions[first], box);
            const Vec3 sourceDelta = minimumImage(sourcePositions[second]-sourcePositions[first], box);
            const double referenceDistance = std::sqrt(referenceDelta.dot(referenceDelta));
            const double sourceDistance = std::sqrt(sourceDelta.dot(sourceDelta));
            const double residual = std::fabs(sourceDistance-referenceDistance);
            if (residual > maximumResidual) {
                maximumResidual = residual;
                toleranceAtMaximum = rigidTolerance(referenceDistance);
            }
        }
    if (maximumResidual > toleranceAtMaximum)
        throw OpenMMException("RelocationMC selected source molecule is not rigid-congruent to its reference conformation; max pair-distance residual="+
                std::to_string(maximumResidual)+" nm, tolerance="+std::to_string(toleranceAtMaximum)+" nm.");
}

/**
 * Compute the proper rotation from a source conformation to the actual accepted
 * target coordinates.  Both coordinate sets are minimum-imaged about their
 * configured reference site.  Pair distances are checked before a caller
 * mutates its Context, using 1e-3 nm absolute plus 1e-4 relative tolerance;
 * fit residual is diagnostic only.
 *
 * A nonlinear molecule uses a proper orthogonal least-squares fit over all local
 * vectors.  A linear molecule uses the minimum-angle rotation between its old
 * and new axes; rotation about that axis is intentionally left unchanged.  A
 * single-site (or fully coincident) molecule has no orientational degree of
 * freedom and returns the identity rotation.
 */
inline RotationMatrix computeRotationBetweenConformations(const std::vector<Vec3>& sourcePositions,
        const std::vector<Vec3>& targetPositions, const Vec3 box[3], int referenceSite) {
    if (targetPositions.size() != sourcePositions.size() || targetPositions.empty())
        throw OpenMMException("RelocationMC source and target conformations must have the same nonzero size.");
    if (referenceSite < 0 || referenceSite >= static_cast<int>(targetPositions.size()))
        throw OpenMMException("RelocationMC reference site is out of range for velocity rotation.");
    validateRigidCongruence(targetPositions, sourcePositions, box);

    int primary = -1;
    double primaryNorm2 = 0.0;
    std::vector<Vec3> targetVectors(targetPositions.size());
    std::vector<Vec3> sourceVectors(sourcePositions.size());
    for (int site = 0; site < static_cast<int>(targetPositions.size()); site++) {
        targetVectors[site] = minimumImage(targetPositions[site]-targetPositions[referenceSite], box);
        sourceVectors[site] = minimumImage(sourcePositions[site]-sourcePositions[referenceSite], box);
        const double norm2 = targetVectors[site].dot(targetVectors[site]);
        if (norm2 > primaryNorm2) {
            primaryNorm2 = norm2;
            primary = site;
        }
    }
    if (primary < 0 || primaryNorm2 <= 1.0e-24)
        return identityRotation();

    int secondary = -1;
    double secondaryArea2 = 0.0;
    for (int site = 0; site < static_cast<int>(targetPositions.size()); site++) {
        const Vec3 area = targetVectors[primary].cross(targetVectors[site]);
        const double area2 = area.dot(area);
        if (area2 > secondaryArea2) {
            secondaryArea2 = area2;
            secondary = site;
        }
    }

    const Vec3 sourceX = normalized(sourceVectors[primary],
            "RelocationMC source molecule collapsed along its primary body axis.");
    const Vec3 targetX = normalized(targetVectors[primary],
            "RelocationMC accepted molecule collapsed along its primary body axis.");
    RotationMatrix rotation;
    if (secondary < 0 || secondaryArea2 <= 1.0e-12*primaryNorm2*primaryNorm2)
        rotation = rotationBetweenUnitVectors(sourceX, targetX);
    else
        rotation = bestFitProperRotation(sourceVectors, targetVectors);

    // For pair-congruent but not mathematically identical shapes, no proper
    // rotation maps every vector exactly.  The least-squares residual is therefore
    // diagnostic, not a second support rejection criterion.  The strict support
    // check is the gross pair-distance validation above; Q itself is proper by
    // construction and minimizes the local-vector residual.
    return rotation;
}

/** Apply v'=V_COM+Q(v-V_COM), using the supplied masses for the COM velocity. */
inline std::vector<Vec3> rotateVelocitiesPreservingCOM(const std::vector<double>& masses,
        const std::vector<Vec3>& velocities, const RotationMatrix& rotation) {
    if (masses.size() != velocities.size() || masses.empty())
        throw OpenMMException("RelocationMC masses and velocities must have the same nonzero size.");
    double totalMass = 0.0;
    Vec3 comVelocity(0.0, 0.0, 0.0);
    for (int site = 0; site < static_cast<int>(masses.size()); site++) {
        if (masses[site] < 0.0)
            throw OpenMMException("RelocationMC particle masses cannot be negative.");
        totalMass += masses[site];
        comVelocity += velocities[site]*masses[site];
    }
    if (!(totalMass > 0.0))
        throw OpenMMException("RelocationMC requires positive total mass for every labile molecule.");
    comVelocity *= 1.0/totalMass;

    std::vector<Vec3> rotated(velocities.size());
    for (int site = 0; site < static_cast<int>(velocities.size()); site++)
        rotated[site] = comVelocity+applyRotation(rotation, velocities[site]-comVelocity);
    return rotated;
}

} // namespace RelocationMCVelocity
} // namespace RelocationMCPlugin
} // namespace OpenMM

#endif
