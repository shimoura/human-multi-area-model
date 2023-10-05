export PYTHONPATH=""

  snakemake --jobs 100 --use-conda --cluster-config cluster.json --cluster "sbatch --account={cluster.account} --job-name={cluster.job-name} --output={cluster.output} --error={cluster.error} --threads-per-core=1 --ntasks={cluster.ntasks} --ntasks-per-node={cluster.ntasks-per-node} --nodes={cluster.nodes} --time={cluster.time} --mail-type=BEGIN,END,FAIL --exclusive --partition={cluster.partition} --export=ALL,SRUN_CPUS_PER_TASK={cluster.cpus-per-task}"
