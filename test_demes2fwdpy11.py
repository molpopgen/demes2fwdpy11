import build_model
import demes
import numpy as np
import unittest
import fwdpy11


def check_valid_demography(cls):
    def _valid_fwdpy11_demography(self):
        try:
            _ = fwdpy11.DemographyDebugger(
                [100] * len(self.demog.metadata["initial_sizes"]), self.demog
            )
        except:
            self.fail("unexpected exception")

    cls.test_validity = _valid_fwdpy11_demography
    return cls


@check_valid_demography
class TestNoEvents(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.g = demes.DemeGraph(
            description="test demography", time_units="generations"
        )
        self.g.deme(id="deme", initial_size=1000)
        self.g.get_demographic_events()
        self.demog = build_model.build_from_deme_graph(self.g, 10)


@check_valid_demography
class TestTwoEpoch(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.g = demes.DemeGraph(
            description="test demography", time_units="generations"
        )
        self.g.deme(
            id="deme",
            epochs=[
                demes.Epoch(initial_size=1000, end_time=100),
                demes.Epoch(initial_size=2000, end_time=0),
            ],
        )
        self.g.get_demographic_events()
        self.demog = build_model.build_from_deme_graph(self.g, 10)

    def test_size_change_params(self):
        self.assertTrue(len(self.demog.model.set_deme_sizes) == 1)
        self.assertTrue(
            self.demog.model.set_deme_sizes[0].when
            == self.demog.metadata["burnin_time"]
        )
        self.assertTrue(self.demog.model.set_deme_sizes[0].new_size == 2000)


@check_valid_demography
class TestNonGenerationUnits(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.g = demes.DemeGraph(
            description="test demography", time_units="years", generation_time=25
        )
        self.g.deme(
            id="Pop",
            epochs=[
                demes.Epoch(initial_size=10000, end_time=25000),
                demes.Epoch(initial_size=1000, final_size=20000, end_time=0),
            ],
        )
        self.g.get_demographic_events()
        self.demog = build_model.build_from_deme_graph(self.g, 10)

    def test_conversion_to_generations(self):
        self.assertTrue(
            self.demog.model.set_deme_sizes[0].when
            == self.demog.metadata["burnin_time"]
        )
        self.assertTrue(
            self.demog.metadata["total_simulation_length"]
            == self.demog.metadata["burnin_time"] + 25000 // 25
        )
        self.assertTrue(self.demog.model.set_deme_sizes[0].new_size == 1000)


@check_valid_demography
class TestSelfingShift(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.g = demes.DemeGraph(
            description="test demography", time_units="generations"
        )
        self.g.deme(
            id="Selfer",
            epochs=[
                demes.Epoch(initial_size=1000, end_time=1000),
                demes.Epoch(initial_size=1000, end_time=0, selfing_rate=0.2),
            ],
        )
        self.g.get_demographic_events()
        self.demog = build_model.build_from_deme_graph(self.g, 10)

    def test_selfing_parameters(self):
        self.assertTrue(
            self.demog.model.set_selfing_rates[0].when
            == self.demog.metadata["burnin_time"]
        )
        self.assertTrue(self.demog.model.set_selfing_rates[0].S == 0.2)


@check_valid_demography
class TestSelfing(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.g = demes.DemeGraph(
            description="test demography", time_units="generations"
        )
        self.g.deme(id="Selfer", initial_size=1000, selfing_rate=0.5)
        self.g.get_demographic_events()
        self.demog = build_model.build_from_deme_graph(self.g, 10)

    def test_single_pop_selfing(self):
        self.assertTrue(self.demog.model.set_selfing_rates[0].when == 0)
        self.assertTrue(self.demog.model.set_selfing_rates[0].S == 0.5)


@check_valid_demography
class TestSplit(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.g = demes.DemeGraph(
            description="test demography", time_units="generations"
        )
        self.g.deme(id="Ancestor", initial_size=1000, end_time=200)
        self.g.deme("Deme1", initial_size=100, ancestors=["Ancestor"])
        self.g.deme("Deme2", initial_size=100, ancestors=["Ancestor"])
        self.g.get_demographic_events()
        self.demog = build_model.build_from_deme_graph(self.g, 10)

    def test_size_changes(self):
        self.assertTrue(len(self.demog.model.set_deme_sizes) == 3)
        self.assertTrue(self.demog.model.set_deme_sizes[0].deme == 0)
        self.assertTrue(self.demog.model.set_deme_sizes[0].new_size == 0)
        self.assertTrue(
            self.demog.model.set_deme_sizes[0].when
            == self.demog.metadata["burnin_time"]
        )
        self.assertTrue(self.demog.model.set_deme_sizes[1].deme == 1)
        self.assertTrue(self.demog.model.set_deme_sizes[1].new_size == 100)
        self.assertTrue(
            self.demog.model.set_deme_sizes[1].when
            == self.demog.metadata["burnin_time"]
        )
        self.assertTrue(self.demog.model.set_deme_sizes[2].deme == 2)
        self.assertTrue(self.demog.model.set_deme_sizes[2].new_size == 100)
        self.assertTrue(
            self.demog.model.set_deme_sizes[2].when
            == self.demog.metadata["burnin_time"]
        )


@check_valid_demography
class TestSplitMigration(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.g = demes.DemeGraph(
            description="test demography", time_units="generations"
        )
        self.g.deme(id="Ancestor", initial_size=1000, end_time=200)
        self.g.deme("Deme1", initial_size=100, ancestors=["Ancestor"])
        self.g.deme("Deme2", initial_size=100, ancestors=["Ancestor"])
        self.g.migration(source="Deme1", dest="Deme2", rate=0.01)
        self.g.migration(source="Deme2", dest="Deme1", rate=0.01)
        self.g.get_demographic_events()
        self.demog = build_model.build_from_deme_graph(self.g, 10)


@check_valid_demography
class TestSplitThreeWay(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.g = demes.DemeGraph(
            description="test demography", time_units="generations"
        )
        self.g.deme(id="Ancestor", initial_size=1000, end_time=200)
        self.g.deme("Deme1", initial_size=100, ancestors=["Ancestor"])
        self.g.deme("Deme2", initial_size=100, ancestors=["Ancestor"])
        self.g.deme("Deme3", initial_size=200, ancestors=["Ancestor"])
        self.g.get_demographic_events()
        self.demog = build_model.build_from_deme_graph(self.g, 10)


@check_valid_demography
class TestBranch(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.g = demes.DemeGraph(description="test branch", time_units="generations")
        self.g.deme(id="Ancestor", initial_size=1000)
        self.g.deme("Deme1", initial_size=100, ancestors=["Ancestor"], start_time=100)
        self.g.get_demographic_events()
        self.demog = build_model.build_from_deme_graph(self.g, 10)


@check_valid_demography
class TestBranchMigration(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.g = demes.DemeGraph(description="test branch", time_units="generations")
        self.g.deme(id="Ancestor", initial_size=1000)
        self.g.deme("Deme1", initial_size=100, ancestors=["Ancestor"], start_time=100)
        self.g.migration(source="Ancestor", dest="Deme1", rate=0.01)
        self.g.migration(source="Deme1", dest="Ancestor", rate=0.01)
        self.g.get_demographic_events()
        self.demog = build_model.build_from_deme_graph(self.g, 10)


@check_valid_demography
class TestMultipleBranches(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.g = demes.DemeGraph(description="test branch", time_units="generations")
        self.g.deme(id="Ancestor", initial_size=1000)
        self.g.deme("Deme1", initial_size=100, ancestors=["Ancestor"], start_time=100)
        self.g.deme("Deme2", initial_size=200, ancestors=["Ancestor"], start_time=50)
        self.g.deme("Deme3", initial_size=300, ancestors=["Deme1"], start_time=20)
        self.g.get_demographic_events()
        self.demog = build_model.build_from_deme_graph(self.g, 10)


@check_valid_demography
class TestSplitsBranches(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.g = demes.DemeGraph(description="test", time_units="generations")
        self.g.deme(id="A", initial_size=1000, end_time=100)
        self.g.deme(id="B", initial_size=1000, ancestors=["A"], start_time=200)
        self.g.deme(id="C", initial_size=1000, ancestors=["A"])
        self.g.deme(id="D", initial_size=1000, ancestors=["A"])
        self.g.get_demographic_events()
        self.demog = build_model.build_from_deme_graph(self.g, 10)


@check_valid_demography
class TestIslandModel(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.g = demes.DemeGraph(description="island", time_units="generations")
        self.g.deme(id="Island1", initial_size=100)
        self.g.deme(id="Island2", initial_size=200)
        self.g.migration(source="Island1", dest="Island2", rate=0.01)
        self.g.migration(source="Island2", dest="Island1", rate=0.02)
        self.g.get_demographic_events()
        self.demog = build_model.build_from_deme_graph(self.g, 10)

    def test_demog_attributes(self):
        self.assertTrue(
            self.demog.metadata["burnin_time"]
            == sum(self.demog.metadata["initial_sizes"].values()) * 10
        )
        self.assertTrue(len(self.demog.model.set_migration_rates) == 0)


@check_valid_demography
class TestIslandModelRateChange(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.g = demes.DemeGraph(description="island", time_units="generations")
        self.g.deme(id="Island1", initial_size=100)
        self.g.deme(id="Island2", initial_size=200)
        self.g.migration(source="Island1", dest="Island2", rate=0.01, end_time=500)
        self.g.migration(source="Island2", dest="Island1", rate=0.02)
        self.g.migration(source="Island1", dest="Island2", rate=0.05, start_time=500)
        self.g.get_demographic_events()
        self.demog = build_model.build_from_deme_graph(self.g, 10)

    def test_burnin_time(self):
        self.assertTrue(
            self.demog.metadata["burnin_time"]
            == sum(self.demog.metadata["initial_sizes"].values()) * 10
        )

    def test_num_mig_rate_changes(self):
        self.assertTrue(len(self.demog.model.set_migration_rates) == 1)

    def test_total_sim_length(self):
        self.assertTrue(
            self.demog.metadata["total_simulation_length"]
            == self.demog.metadata["burnin_time"] + 500
        )


@check_valid_demography
class TestTwoPopMerger(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.g = demes.DemeGraph(
            description="split then merger", time_units="generations"
        )
        self.g.deme(id="Ancestral", initial_size=1000, end_time=1000)
        self.g.deme(
            id="Parent1", initial_size=500, end_time=500, ancestors=["Ancestral"]
        )
        self.g.deme(
            id="Parent2", initial_size=500, end_time=500, ancestors=["Ancestral"]
        )
        self.g.deme(
            id="Child",
            initial_size=1000,
            ancestors=["Parent1", "Parent2"],
            proportions=[0.5, 0.5],
            start_time=500,
        )
        self.g.get_demographic_events()
        self.demog = build_model.build_from_deme_graph(self.g, 10)

    def test_total_sim_length(self):
        self.assertTrue(
            self.demog.metadata["total_simulation_length"]
            == self.demog.metadata["burnin_time"] + 1000
        )

    def test_num_size_changes(self):
        self.assertTrue(len(self.demog.model.set_deme_sizes) == 6)


@check_valid_demography
class TestFourWayMerger(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.g = demes.DemeGraph(
            description="split then merger", time_units="generations"
        )
        self.g.deme(id="Ancestral", initial_size=1000, end_time=1000)
        self.g.deme(id="A", initial_size=500, end_time=700, ancestors=["Ancestral"])
        self.g.deme(id="B", initial_size=500, end_time=500, ancestors=["Ancestral"])
        self.g.deme(id="Parent1", initial_size=200, end_time=100, ancestors=["A"])
        self.g.deme(id="Parent2", initial_size=300, end_time=100, ancestors=["A"])
        self.g.deme(id="Parent3", initial_size=100, end_time=100, ancestors=["B"])
        self.g.deme(id="Parent4", initial_size=400, end_time=100, ancestors=["B"])
        self.g.deme(
            id="Child",
            initial_size=1000,
            ancestors=["Parent1", "Parent2", "Parent3", "Parent4"],
            proportions=[0.2, 0.3, 0.1, 0.4],
            start_time=100,
        )
        self.g.get_demographic_events()
        self.demog = build_model.build_from_deme_graph(self.g, 10)

    def test_total_sim_length(self):
        self.assertTrue(
            self.demog.metadata["total_simulation_length"]
            == self.demog.metadata["burnin_time"] + 1000
        )

    def test_num_size_changes(self):
        self.assertTrue(len(self.demog.model.set_deme_sizes) == 14)



