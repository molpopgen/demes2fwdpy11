import build_model
import demes
import numpy as np
import unittest
import fwdpy11


class TestSingleDeme(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.g = demes.DemeGraph(
            description="test demography", time_units="generations"
        )

    def valid_fwdpy11_demography(self):
        self.g.get_demographic_events()
        self.demog = build_model.build_from_deme_graph(self.g, 10)
        try:
            _ = fwdpy11.DemographyDebugger([100], self.demog)
        except:
            self.fail("unexpected exception")

    def test_constant_size(self):
        self.g.deme(id="deme", initial_size=1000)
        self.valid_fwdpy11_demography()

    def test_two_epoch_model(self):
        self.g.deme(
            id="deme",
            epochs=[
                demes.Epoch(initial_size=1000, end_time=100),
                demes.Epoch(initial_size=2000, end_time=0),
            ],
        )
        self.valid_fwdpy11_demography()
        self.assertTrue(len(self.demog.model.set_deme_sizes) == 1)
        self.assertTrue(
            self.demog.model.set_deme_sizes[0].when
            == self.demog.metadata["burnin_time"]
        )
        self.assertTrue(self.demog.model.set_deme_sizes[0].new_size == 2000)


class TestNonGenerationUnits(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.g = demes.DemeGraph(
            description="test demography", time_units="years", generation_time=25
        )

    def valid_fwdpy11_demography(self):
        self.g.get_demographic_events()
        self.demog = build_model.build_from_deme_graph(self.g, 10)
        try:
            _ = fwdpy11.DemographyDebugger([100], self.demog)
        except:
            self.fail("unexpected exception")

    def test_single_deme_years(self):
        self.g.deme(
            id="Pop",
            epochs=[
                demes.Epoch(initial_size=10000, end_time=25000),
                demes.Epoch(initial_size=1000, final_size=20000, end_time=0),
            ],
        )
        self.valid_fwdpy11_demography()
        self.assertTrue(
            self.demog.model.set_deme_sizes[0].when
            == self.demog.metadata["burnin_time"]
        )
        self.assertTrue(
            self.demog.metadata["total_simulation_length"]
            == self.demog.metadata["burnin_time"] + 25000 // 25
        )
        self.assertTrue(self.demog.model.set_deme_sizes[0].new_size == 1000)


class TestSelfing(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.g = demes.DemeGraph(
            description="test demography", time_units="generations"
        )

    def valid_fwdpy11_demography(self):
        self.g.get_demographic_events()
        self.demog = build_model.build_from_deme_graph(self.g, 10)
        try:
            _ = fwdpy11.DemographyDebugger([100], self.demog)
        except:
            self.fail("unexpected exception")

    def test_single_pop_selfing_shift(self):
        self.g.deme(
            id="Selfer",
            epochs=[
                demes.Epoch(initial_size=1000, end_time=1000),
                demes.Epoch(initial_size=1000, end_time=0, selfing_rate=0.2),
            ],
        )
        self.valid_fwdpy11_demography()
        self.assertTrue(
            self.demog.model.set_selfing_rates[0].when
            == self.demog.metadata["burnin_time"]
        )
        self.assertTrue(self.demog.model.set_selfing_rates[0].S == 0.2)

    def test_single_pop_selfing(self):
        self.g.deme(id="Selfer", initial_size=1000, selfing_rate=0.5)
        self.valid_fwdpy11_demography()
        self.assertTrue(self.demog.model.set_selfing_rates[0].when == 0)
        self.assertTrue(self.demog.model.set_selfing_rates[0].S == 0.5)


class TestSplits(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.g = demes.DemeGraph(
            description="test demography", time_units="generations"
        )
        self.g.deme(id="Ancestor", initial_size=1000, end_time=200)
        self.g.deme("Deme1", initial_size=100, ancestors=["Ancestor"])
        self.g.deme("Deme2", initial_size=100, ancestors=["Ancestor"])

    def valid_fwdpy11_demography(self):
        self.g.get_demographic_events()
        self.demog = build_model.build_from_deme_graph(self.g, 10)
        try:
            _ = fwdpy11.DemographyDebugger([100], self.demog)
        except:
            self.fail("unexpected exception")

    def test_split_no_migration(self):
        self.valid_fwdpy11_demography()

    def test_split_with_migration(self):
        self.g.migration(source="Deme1", dest="Deme2", rate=0.01)
        self.g.migration(source="Deme2", dest="Deme1", rate=0.01)
        self.valid_fwdpy11_demography()

    def test_three_way_split(self):
        self.g.deme("Deme3", initial_size=200, ancestors=["Ancestor"])
        self.valid_fwdpy11_demography()


class TestBranches(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.g = demes.DemeGraph(description="test split", time_units="generations")
        self.g.deme(id="Ancestor", initial_size=1000)
        self.g.deme("Deme1", initial_size=100, ancestors=["Ancestor"], start_time=100)

    def valid_fwdpy11_demography(self):
        self.g.get_demographic_events()
        self.demog = build_model.build_from_deme_graph(self.g, 10)
        try:
            _ = fwdpy11.DemographyDebugger([100], self.demog)
        except:
            self.fail("unexpected exception")

    def test_branch(self):
        self.valid_fwdpy11_demography()

    def test_branch_with_migration(self):
        self.g.migration(source="Ancestor", dest="Deme1", rate=0.01)
        self.g.migration(source="Deme1", dest="Ancestor", rate=0.01)
        self.valid_fwdpy11_demography()
