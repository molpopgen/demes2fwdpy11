import build_model
import demes
import numpy as np
import unittest


class TestFwdpy11Conversion(unittest.TestCase):
    def test_single_constant_size(self):
        g = demes.DemeGraph(description="test", time_units="generations")
        g.deme(id="Pop", initial_size=1000)
        demog = build_model.build_from_deme_graph(g, 10)
        self.assertTrue(
            demog.metadata["burnin_time"] == demog.metadata["total_simulation_length"]
        )
        self.assertTrue(len(demog.metadata["deme_labels"]) == 1)
        self.assertTrue(demog.metadata["initial_sizes"][0] == 1000)

    def test_two_epoch_model(self):
        g = demes.DemeGraph(description="test", time_units="generations")
        g.deme(
            id="Pop",
            epochs=[
                demes.Epoch(initial_size=1000, end_time=100),
                demes.Epoch(initial_size=2000, end_time=0),
            ],
        )
        demog = build_model.build_from_deme_graph(g, 10)
        self.assertTrue(len(demog.model.set_deme_sizes) == 1)
        self.assertTrue(
            demog.model.set_deme_sizes[0].when == demog.metadata["burnin_time"]
        )
        self.assertTrue(demog.model.set_deme_sizes[0].new_size == 2000)

    def test_single_deme_years(self):
        g = demes.DemeGraph(description="test", time_units="years", generation_time=25)
        g.deme(
            id="Pop",
            epochs=[
                demes.Epoch(initial_size=10000, end_time=25000),
                demes.Epoch(initial_size=1000, final_size=20000, end_time=0),
            ],
        )
        demog = build_model.build_from_deme_graph(g, 10)
        self.assertTrue(
            demog.model.set_deme_sizes[0].when == demog.metadata["burnin_time"]
        )
        self.assertTrue(
            demog.metadata["total_simulation_length"]
            == demog.metadata["burnin_time"] + 25000 / 25
        )
        self.assertTrue(demog.model.set_deme_sizes[0].new_size == 1000)

    def test_single_pop_selfing(self):
        g = demes.DemeGraph(
            description="test selfing rate change", time_units="generations"
        )
        g.deme(
            id="Selfer",
            epochs=[
                demes.Epoch(initial_size=1000, end_time=1000),
                demes.Epoch(initial_size=1000, end_time=0, selfing_rate=0.2),
            ],
        )
        demog = build_model.build_from_deme_graph(g, 10)
        self.assertTrue(
            demog.model.set_selfing_rates[0].when == demog.metadata["burnin_time"]
        )
        self.assertTrue(demog.model.set_selfing_rates[0].S == 0.2)
        # model with selfing rate set for all time
        g = demes.DemeGraph(
            description="test selfing rate change", time_units="generations"
        )
        g.deme(id="Selfer", initial_size=1000, selfing_rate=0.5)
        demog = build_model.build_from_deme_graph(g, 10)
        self.assertTrue(demog.model.set_selfing_rates[0].when == 0)
        self.assertTrue(demog.model.set_selfing_rates[0].S == 0.5)

    def test_split_no_migration(self):
        g = demes.DemeGraph(description="test split", time_units="generations")
        g.deme(id="Ancestor", initial_size=1000, end_time=200)
        g.deme("Deme1", initial_size=100, ancestors=["Ancestor"])
        g.deme("Deme2", initial_size=100, ancestors=["Ancestor"])
        g.get_demographic_events()
        demog = build_model.build_from_deme_graph(g, 10)
        self.assertTrue(len(demog.model.set_migration_rates) == 4)
        self.assertTrue(
            demog.model.set_migration_rates[0].when == demog.metadata["burnin_time"]
        )
        self.assertTrue(
            demog.model.set_migration_rates[1].when == demog.metadata["burnin_time"]
        )
        self.assertTrue(
            demog.model.set_migration_rates[2].when == demog.metadata["burnin_time"] + 1
        )
        self.assertTrue(
            demog.model.set_migration_rates[3].when == demog.metadata["burnin_time"] + 1
        )

    def test_split_with_migration(self):
        g = demes.DemeGraph(description="test split", time_units="generations")
        g.deme(id="Ancestor", initial_size=1000, end_time=200)
        g.deme("Deme1", initial_size=100, ancestors=["Ancestor"])
        g.deme("Deme2", initial_size=100, ancestors=["Ancestor"])
        g.migrations = [
            # asymmetric migration from Deme1 to to Deme2
            # starts from time of split and ends before present
            demes.Migration(
                source="Deme1", dest="Deme2", rate=0.01, start_time=200, end_time=100
            )
        ]
        g.get_demographic_events()
        demog = build_model.build_from_deme_graph(g, 10)
        self.assertTrue(
            demog.model.set_migration_rates[-1].when
            == demog.metadata["burnin_time"] + 100
        )
        self.assertTrue(len(demog.model.set_migration_rates) == 5)

    def test_three_way_split(self):
        g = demes.DemeGraph(description="test split", time_units="generations")
        g.deme(id="Ancestor", initial_size=1000, end_time=200)
        g.deme("Deme1", initial_size=100, ancestors=["Ancestor"])
        g.deme("Deme2", initial_size=100, ancestors=["Ancestor"])
        g.deme("Deme3", initial_size=200, ancestors=["Ancestor"])
        g.get_demographic_events()
        demog = build_model.build_from_deme_graph(g, 10)
        self.assertTrue(len(demog.model.set_migration_rates) == 6)
        self.assertTrue(
            demog.model.set_migration_rates[0].when == demog.metadata["burnin_time"]
        )
        self.assertTrue(
            demog.model.set_migration_rates[2].when == demog.metadata["burnin_time"]
        )
        self.assertTrue(
            demog.model.set_migration_rates[3].when == demog.metadata["burnin_time"] + 1
        )
        self.assertTrue(
            demog.model.set_migration_rates[5].when == demog.metadata["burnin_time"] + 1
        )

    def test_branch(self):
        g = demes.DemeGraph(description="test split", time_units="generations")
        g.deme(id="Ancestor", initial_size=1000)
        g.deme("Deme1", initial_size=100, ancestors=["Ancestor"], start_time=100)
        g.get_demographic_events()
        demog = build_model.build_from_deme_graph(g, 10)
        self.assertTrue(len(demog.model.set_migration_rates) == 2)
        self.assertTrue(
            demog.model.set_migration_rates[0].when == demog.metadata["burnin_time"]
        )
        self.assertTrue(
            demog.model.set_migration_rates[1].when == demog.metadata["burnin_time"] + 1
        )
        self.assertTrue(
            np.all(demog.model.set_migration_rates[0].migrates == np.array([1, 0]))
        )
        self.assertTrue(
            np.all(demog.model.set_migration_rates[1].migrates == np.array([0, 1]))
        )

    def test_branch_with_migration(self):
        g = demes.DemeGraph(description="test split", time_units="generations")
        g.deme(id="Ancestor", initial_size=1000)
        g.deme("Deme1", initial_size=100, ancestors=["Ancestor"], start_time=100)
        g.migrations = [
            demes.Migration(
                source="Ancestor", dest="Deme1", rate=0.01, start_time=100, end_time=0
            ),
            demes.Migration(
                source="Deme1", dest="Ancestor", rate=0.01, start_time=100, end_time=0
            ),
        ]
        g.get_demographic_events()
        demog = build_model.build_from_deme_graph(g, 10)
        self.assertTrue(len(demog.model.set_migration_rates) == 3)
        self.assertTrue(
            demog.model.set_migration_rates[0].when == demog.metadata["burnin_time"]
        )
        self.assertTrue(
            demog.model.set_migration_rates[2].when == demog.metadata["burnin_time"] + 1
        )
        self.assertTrue(
            np.all(demog.model.set_migration_rates[1].migrates == np.array([1, 0]))
        )
        self.assertTrue(
            np.all(
                demog.model.set_migration_rates[2].migrates == np.array([0.01, 0.99])
            )
        )
