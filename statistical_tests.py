import fwdpy11
import build_model
import moments
import demes, demes.convert
import numpy as np

# from a YAML file, build the fwdpy11 demography, then compare neutral SFS to
# moments simulation from the same YAML file

pop_ids = ["deme1", "deme2"]
sample_sizes = [20, 20]

## moments simulation
dg = demes.load("test_demog.yaml")
sfs_moments = demes.convert.SFS(dg, pop_ids, sample_sizes)

## fwdpy11 simulation
demog = build_model.build_from_yaml("test_demog.yaml", 10)
# set up the rest of the simulation parameters
recregions = [fwdpy11.PoissonInterval(beg=0.0, end=1.0, mean=1.0)]
pdict = {
    "nregions": [],
    "sregions": [],
    "recregions": recregions,
    "rates": (0.0, 0.0, None),
    "gvalue": fwdpy11.Additive(scaling=2.0),
    "simlen": demog.metadata["total_simulation_length"],
    "demography": demog.model,
}
params = fwdpy11.ModelParams(**pdict)
# run the simulation
rng = fwdpy11.GSLrng(42)
pop = fwdpy11.DiploidPopulation(demog.metadata["initial_sizes"][0], 1.0)
fwdpy11.evolvets(rng=rng, pop=pop, params=params, simplification_interval=100)
assert pop.generation == demog.metadata["total_simulation_length"]
# get the frequency spectrum
nodes = np.array(pop.tables.nodes, copy=False)
alive_nodes = pop.alive_nodes
deme0_nodes = alive_nodes[np.where(nodes["deme"][alive_nodes] == 1)[0]]
deme1_nodes = alive_nodes[np.where(nodes["deme"][alive_nodes] == 2)[0]]
assert len(deme0_nodes) == 2 * 4000
assert len(deme1_nodes) == 2 * 20000
# add mutations to trees
theta = 2.0
nmuts = fwdpy11.infinite_sites(rng, pop, theta)
print("number of mutations:", nmuts)
# compute SFS
sfs_fwdpy11 = pop.tables.fs([deme0_nodes[:sample_sizes[0]], deme1_nodes[:sample_sizes[1]]])

## rescale moments SFS
sfs_moments *= 4 * theta * demog.metadata["initial_sizes"][0]

sfs_fwdpy11 = moments.Spectrum(sfs_fwdpy11.todense())

moments.Plotting.plot_2d_comp_Poisson(sfs_moments, sfs_fwdpy11)
