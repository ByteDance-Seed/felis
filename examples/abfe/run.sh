set -ex

export RDMAV_FORK_SAFE=1
export OMP_NUM_THREADS=1
export OPENMM_CPU_THREADS=1
git_repo_dir=$(git rev-parse --show-toplevel)
export PYTHONPATH=$git_repo_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# prepare tyk2 input
rm -rf input/
mkdir input/
tyk2_dir=$git_repo_dir/pl_bfe_dataset/Schrodinger/jacs/tyk2/
cp $tyk2_dir/protein_ff14sb/protein_w_cofactors.gro input/
cp $tyk2_dir/protein_ff14sb/protein_w_cofactors.top input/
cp $tyk2_dir/ligands/ejm_31.sdf input/
cp $tyk2_dir/ligands_itps_joint-25/ejm_31.itp input/

# run abfe
python3 -m felis.app.abfe --abfecfg abfecfg.yaml