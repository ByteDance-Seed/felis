// Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: LGPL-3.0-or-later

%module PerturbNonbondedPlugin

%import(module="openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include <std_string.i>
%include <std_vector.i>
%include <std_map.i>

namespace std {
    %template(vectord) vector<double>;
    %template(vectori) vector<int>;
    %template(FloatVector) vector<float>;
    %template(StringVector) vector<string>;
    %template(ConstCharVector) vector<const char*>;
};

%{
#include "PerturbNonbondedForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
#include <vector>
%}

%pythoncode %{
import openmm as mm
import openmm.unit as unit
%}

/* Convert C++ exceptions to Python exceptions. */
%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
        return NULL;
    }
}


namespace PerturbNonbondedPlugin {

class PerturbNonbondedForce : public OpenMM::Force {
public:
   enum PerturbNonbondedMethod {
      NoCutoff =  0,
      CutoffNonPeriodic =  1,
      CutoffPeriodic =  2,
      Ewald =  3,
      PME =  4,
      LJPME =  5
   };

   PerturbNonbondedForce() ;

   int getNumParticles() const ;
   int getNumExceptions() const ;
   int getNumGlobalParameters() const ;
   int getNumParticleParameterOffsets() const ;
   int getNumExceptionParameterOffsets() const ;
   PerturbNonbondedMethod getNonbondedMethod() const ;
   void setNonbondedMethod(PerturbNonbondedMethod method) ;
   double getCutoffDistance() const ;
   void setCutoffDistance(double distance) ;
   bool getUseSwitchingFunction() const ;
   void setUseSwitchingFunction(bool use) ;
   double getSwitchingDistance() const ;
   void setSwitchingDistance(double distance) ;
   double getReactionFieldDielectric() const ;
   void setReactionFieldDielectric(double dielectric) ;
   double getEwaldErrorTolerance() const ;
   void setEwaldErrorTolerance(double tol) ;
   %apply double & OUTPUT { double & alpha };
   %apply int & OUTPUT { int & nx };
   %apply int & OUTPUT { int & ny };
   %apply int & OUTPUT { int & nz };
   void getPMEParameters(double &alpha, int &nx, int &ny, int &nz) const ;
   %clear double & alpha;
   %clear int & nx;
   %clear int & ny;
   %clear int & nz;
   %apply double & OUTPUT { double & alpha };
   %apply int & OUTPUT { int & nx };
   %apply int & OUTPUT { int & ny };
   %apply int & OUTPUT { int & nz };
   void getLJPMEParameters(double &alpha, int &nx, int &ny, int &nz) const ;
   %clear double & alpha;
   %clear int & nx;
   %clear int & ny;
   %clear int & nz;
   void setPMEParameters(double alpha, int nx, int ny, int nz) ;
   void setLJPMEParameters(double alpha, int nx, int ny, int nz) ;
   %apply double & OUTPUT { double & alpha };
   %apply int & OUTPUT { int & nx };
   %apply int & OUTPUT { int & ny };
   %apply int & OUTPUT { int & nz };
   void getPMEParametersInContext(const OpenMM::Context &context, double &alpha, int &nx, int &ny, int &nz) const ;
   %clear double & alpha;
   %clear int & nx;
   %clear int & ny;
   %clear int & nz;
   %apply double & OUTPUT { double & alpha };
   %apply int & OUTPUT { int & nx };
   %apply int & OUTPUT { int & ny };
   %apply int & OUTPUT { int & nz };
   void getLJPMEParametersInContext(const OpenMM::Context &context, double &alpha, int &nx, int &ny, int &nz) const ;
   %clear double & alpha;
   %clear int & nx;
   %clear int & ny;
   %clear int & nz;
   int addParticle(double chargeA, double sigmaA, double epsilonA, double chargeB, double sigmaB, double epsilonB, bool isDummy, bool isREST2) ;
   %apply double & OUTPUT { double & chargeA };
   %apply double & OUTPUT { double & sigmaA };
   %apply double & OUTPUT { double & epsilonA };
   %apply double & OUTPUT { double & chargeB };
   %apply double & OUTPUT { double & sigmaB };
   %apply double & OUTPUT { double & epsilonB };
   %apply bool & OUTPUT { bool & isDummy };
   %apply bool & OUTPUT { bool & isREST2 };
   void getParticleParameters(int index, double &chargeA, double &sigmaA, double &epsilonA, double &chargeB, double &sigmaB, double &epsilonB, bool &isDummy, bool &isREST2) const ;
   %clear double & chargeA;
   %clear double & sigmaA;
   %clear double & epsilonA;
   %clear double & chargeB;
   %clear double & sigmaB;
   %clear double & epsilonB;
   %clear bool & isDummy;
   %clear bool & isREST2;
   void setParticleParameters(int index, double chargeA, double sigmaA, double epsilonA, double chargeB, double sigmaB, double epsilonB, bool isDummy, bool isREST2) ;
   int addException(int particle1, int particle2, double chargeProd, double sigma, double epsilon, bool replace=false) ;
   %apply int & OUTPUT { int & particle1 };
   %apply int & OUTPUT { int & particle2 };
   %apply double & OUTPUT { double & chargeProd };
   %apply double & OUTPUT { double & sigma };
   %apply double & OUTPUT { double & epsilon };
   void getExceptionParameters(int index, int &particle1, int &particle2, double &chargeProd, double &sigma, double &epsilon) const ;
   %clear int & particle1;
   %clear int & particle2;
   %clear double & chargeProd;
   %clear double & sigma;
   %clear double & epsilon;
   void setExceptionParameters(int index, int particle1, int particle2, double chargeProd, double sigma, double epsilon) ;
   void createExceptionsFromBonds(const std::vector< std::pair< int, int > > &bonds, double coulomb14Scale, double lj14Scale) ;
   int addGlobalParameter(const std::string &name, double defaultValue) ;
   const std::string& getGlobalParameterName(int index) const ;
   void setGlobalParameterName(int index, const std::string &name) ;
   double getGlobalParameterDefaultValue(int index) const ;
   void setGlobalParameterDefaultValue(int index, double defaultValue) ;
   int addParticleParameterOffset(const std::string &parameter, int particleIndex, double chargeScale, double sigmaScale, double epsilonScale) ;
   %apply std::string & OUTPUT { std::string & parameter };
   %apply int & OUTPUT { int & particleIndex };
   %apply double & OUTPUT { double & chargeScale };
   %apply double & OUTPUT { double & sigmaScale };
   %apply double & OUTPUT { double & epsilonScale };
   void getParticleParameterOffset(int index, std::string &parameter, int &particleIndex, double &chargeScale, double &sigmaScale, double &epsilonScale) const ;
   %clear std::string & parameter;
   %clear int & particleIndex;
   %clear double & chargeScale;
   %clear double & sigmaScale;
   %clear double & epsilonScale;
   void setParticleParameterOffset(int index, const std::string &parameter, int particleIndex, double chargeScale, double sigmaScale, double epsilonScale) ;
   int addExceptionParameterOffset(const std::string &parameter, int exceptionIndex, double chargeProdScale, double sigmaScale, double epsilonScale) ;
   %apply std::string & OUTPUT { std::string & parameter };
   %apply int & OUTPUT { int & exceptionIndex };
   %apply double & OUTPUT { double & chargeProdScale };
   %apply double & OUTPUT { double & sigmaScale };
   %apply double & OUTPUT { double & epsilonScale };
   void getExceptionParameterOffset(int index, std::string &parameter, int &exceptionIndex, double &chargeProdScale, double &sigmaScale, double &epsilonScale) const ;
   %clear std::string & parameter;
   %clear int & exceptionIndex;
   %clear double & chargeProdScale;
   %clear double & sigmaScale;
   %clear double & epsilonScale;
   void setExceptionParameterOffset(int index, const std::string &parameter, int exceptionIndex, double chargeProdScale, double sigmaScale, double epsilonScale) ;
   bool getUseDispersionCorrection() const ;
   void setUseDispersionCorrection(bool useCorrection) ;
   int getReciprocalSpaceForceGroup() const ;
   void setReciprocalSpaceForceGroup(int group) ;
   bool getIncludeDirectSpace() const ;
   void setIncludeDirectSpace(bool include) ;
   void updateParametersInContext(OpenMM::Context &context) ;
   virtual bool usesPeriodicBoundaryConditions() const ;
   bool getExceptionsUsePeriodicBoundaryConditions() const ;
   void setExceptionsUsePeriodicBoundaryConditions(bool periodic) ;
   double getSoftcoreAlpha() const ;
   void setSoftcoreAlpha(double alpha) ;

    void setLJForceGroup(int ljForceGroup);
    int getLJForceGroup() const;

    void setDirectSpaceForceGroup(int directSpaceForceGroup);
    int getDirectSpaceForceGroup() const;

    void setIncludeLJ(bool includeLJ);
    bool getIncludeLJ() const;

    void enableEnergyConstantCache(bool cache);
    bool getEnableEnergyConstantCache() const;

    void setLambdaSchedules(
        const std::vector<double>& lambdaA_charge_schedule, 
        const std::vector<double>& lambdaB_charge_schedule,
        const std::vector<double>& lambdaC_charge_schedule,
        const std::vector<double>& lambdaA_vdw_schedule,
        const std::vector<double>& lambdaB_vdw_schedule,
        const std::vector<double>& rest2_scale_schedule
    );
};

%extend PerturbNonbondedForce {
    static PerturbNonbondedPlugin::PerturbNonbondedForce& cast(OpenMM::Force& force) {
        return dynamic_cast<PerturbNonbondedPlugin::PerturbNonbondedForce&>(force);
    }

    static bool isinstance(OpenMM::Force& force) {
        return (dynamic_cast<PerturbNonbondedPlugin::PerturbNonbondedForce*>(&force) != NULL);
    }

  %pythoncode %{
    def addParticle_usingRVdw(self, charge, rVDW, epsilon):
        """Add particle using elemetrary charge.  Rvdw and epsilon,
           which is consistent with AMBER parameter file usage.
           Note that the sum of the radii of the two interacting atoms is
           the minimum energy point in the Lennard Jones potential and
           is often called rMin.  The conversion from sigma follows:
           rVDW = 2^1/6 * sigma/2
        """
        return self.addParticle(charge, rVDW/RVDW_PER_SIGMA, epsilon)

    def addException_usingRMin(self, particle1, particle2,
                               chargeProd, rMin, epsilon):
        """Add interaction exception using the product of the two atoms'
           elementary charges, rMin and epsilon, which is standard for AMBER
           force fields.  Note that rMin is the minimum energy point in the
           Lennard Jones potential.  The conversion from sigma is:
           rMin = 2^1/6 * sigma.
        """
        return self.addException(particle1, particle2,
                                 chargeProd, rMin/RMIN_PER_SIGMA, epsilon)
  %}
}

}
