import argparse
import math
import sys
import typing
import warnings

import attr
import demes
import fwdpy11
import fwdpy11.class_decorators
import fwdpy11.demographic_models
import numpy as np


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--yaml", type=str, help="Demographic model (YAML)")
    parser.add_argument(
        "--burnin",
        type=int,
        help="Burnin time. An integer that will be a multiple of the ancestral population size.",
        default=10,
    )

    return parser


def _valid_time_unit(self, attribute, value) -> None:
    if value not in ["years", "generations"]:
        raise ValueError(f"{attribute.name} must be years or generations")


@attr.s(auto_attribs=True)
class _TimeConverter(object):
    """
    A callable class that'll convert input
    times (int whatever units) to output
    times in generations.
    """

    time_units: str = attr.ib(validator=_valid_time_unit)
    generation_time: demes.demes.Time = attr.ib(
        validator=[demes.demes.positive, demes.demes.finite]
    )

    def __attrs_post_init__(self):
        self.converter = self._build_time_unit_converter()

    def _build_time_unit_converter(self) -> typing.Callable[[demes.demes.Time], int]:
        if self.time_units == "years":
            return lambda x: x / self.generation_time
        elif self.time_units == "generations":
            return lambda x: x

    def __call__(self, input_time: demes.demes.Time) -> int:
        return np.rint(self.converter(input_time)).astype(int)


@fwdpy11.class_decorators.attr_class_to_from_dict
@attr.s(auto_attribs=True)
class _Fwdpy11Events(object):
    """
    One stop shop for adding things we support in future versions.

    This class creates some redundancy with fwdpy11.DiscreteDemography,
    but that class is "frozen", so we need a mutable version.
    """

    mass_migrations: typing.List[fwdpy11.MassMigration] = attr.Factory(list)
    set_deme_sizes: typing.List[fwdpy11.SetDemeSize] = attr.Factory(list)
    set_growth_rates: typing.List[fwdpy11.SetExponentialGrowth] = attr.Factory(list)
    set_selfing_rates: typing.List[fwdpy11.SetSelfingRate] = attr.Factory(list)
    set_migration_rates: typing.List[fwdpy11.SetMigrationRates] = attr.Factory(list)
    migmatrix: typing.Optional[fwdpy11.MigrationMatrix] = None

    def build_model(self) -> fwdpy11.DiscreteDemography:
        return fwdpy11.DiscreteDemography(**self.asdict())


def _build_deme_id_to_int_map(dg: demes.DemeGraph) -> typing.Dict:
    """
    Convert the string input ID to output integer values.

    For sanity, the output values will be in increasing
    order of "deme origin" times.

    We rely on the epoch times being strictly sorted
    past-to-present in the YAML.  If there are ties,
    populations are sorted lexically by ID within
    start_time.
    """
    temp = []
    for deme in dg.demes:
        assert len(deme.epochs) > 0
        temp.append((deme.epochs[0].start_time, deme.id))

    temp = sorted(temp, key=lambda x: (-x[0], x[1]))

    return {j[1]: i for i, j in enumerate(temp)}


def _get_initial_deme_sizes(dg: demes.DemeGraph, idmap: typing.Dict) -> typing.Dict:
    """
    Build a map of a deme's integer label to its size
    at the start of the simulation for all demes whose
    start_time equals inf.
    """
    otime = _get_most_ancient_deme_start_time(dg)
    rv = dict()
    for deme in dg.demes:
        if deme.epochs[0].start_time == otime:
            rv[idmap[deme.id]] = deme.epochs[0].initial_size

    if len(rv) == 0:
        raise RuntimeError("could not determine initial deme sizes")

    return rv


def _get_most_ancient_deme_start_time(dg: demes.DemeGraph) -> demes.demes.Time:
    return max([e.start_time for d in dg.demes for e in d.epochs])


def _get_most_recent_deme_end_time(dg: demes.DemeGraph) -> demes.demes.Time:
    return min([e.end_time for d in dg.demes for e in d.epochs])


def _get_ancestral_population_size(dg: demes.DemeGraph) -> int:
    """
    Need this for the burnin time.

    If there are > 1 demes with the same most ancient start_time,
    then the ancestral size is considered to be the size
    of all those demes (size of ancestral metapopulation).
    """
    oldest_deme_time = _get_most_ancient_deme_start_time(dg)

    rv = sum(
        [
            e.initial_size
            for d in dg.demes
            for e in d.epochs
            if e.start_time == oldest_deme_time
        ]
    )
    if rv == 0:
        raise RuntimeError("could not determinine ancestral metapopulation size")
    return rv


def _set_initial_migration_matrix(
    dg: demes.DemeGraph, idmap: typing.Dict, events: _Fwdpy11Events
) -> None:
    """
    If there are no migrations nor pulses in dg,
    then the model can use None for migmatrix.
    Otherwise, we need to set it.

    However, there's a complication here. Admixture
    events are given as backwards-in-time ancestry
    proportions.  We have two ways to implement them:

    1. As a fwdpy11.MassMigration event, which
       copies a fraction of the ancestral deme
       to the new deme.  Thus, to get the new
       deme's ancestry proportion right, we
       need to know exact sizes.

    2. As a single-generation change in the migration rate.
       In the simplest case, this works nicely.  However,
       there is a corner case when the migration rates
       are set for the model in the generation immediately
       after the mass migration. fwdpy11 doesn't allow
       you to set the migration rates into a deme > 1
       time in a given generation.
    """
    pass


# FIXME: we aren't processing all of the verbs that we need, but
# we need more YAML examples first.
# migrations (setting migration rates)
# The following are all flavors of mass migration:
# pulses
# splits
# branches
# admixtures
# mergers


def _process_epoch(e: demes.Epoch, idmap: typing.Dict, events: _Fwdpy11Events) -> None:
    """
    Can change sizes, cloning rates, and selfing rates.

    Since fwdpy11 currently doesn't understand cloning, we need
    to raise an error if the rate is not None or nonzero.
    """
    pass


def _process_all_epochs(dg, idmap, events):
    for deme in dg.demes:
        for e in deme.epochs:
            _process_epoch(e, idmap, events)


def _process_migration(
    m: demes.Migration, idmap: typing.Dict, events: _Fwdpy11Events
) -> None:
    pass


def _process_pulse(p: demes.Pulse, idmap: typing.Dict, events: _Fwdpy11Events) -> None:
    pass


def _build_from_deme_graph(
    dg: demes.DemeGraph, burnin: int, source: typing.Optional[typing.Dict] = None
) -> fwdpy11.demographic_models.DemographicModelDetails:
    """
    The workhorse.
    """
    time_converter = _TimeConverter(dg.time_units, dg.generation_time)
    idmap = _build_deme_id_to_int_map(dg)
    initial_sizes = _get_initial_deme_sizes(dg, idmap)
    Nref = _get_ancestral_population_size(dg)
    most_ancient_deme_start = _get_most_ancient_deme_start_time(dg)
    most_recent_deme_end = _get_most_recent_deme_end_time(dg)

    events = _Fwdpy11Events()

    _process_all_epochs(dg, idmap, events)

    if dg.doi != "None":
        doi = dg.doi
    else:
        doi = None

    return fwdpy11.demographic_models.DemographicModelDetails(
        model=events.build_model(),
        name=dg.description,
        source=source,
        parameters=None,
        citation=fwdpy11.demographic_models.DemographicModelCitation(
            DOI=doi, full_citation=None, metadata=None
        ),
        metadata={
            "deme_labels": {j: i for i, j in idmap.items()},
            "initial_sizes": initial_sizes,
        },
    )


def build_from_yaml(
    filename: typing.Text, burnin: int
) -> fwdpy11.demographic_models.DemographicModelDetails:
    """
    Candidate for user-facing function.
    Could also be a part of the DiscreteDemography
    public interface, although static functions are odd in Python?
    """
    dg = demes.load(filename)
    return _build_from_deme_graph(dg, burnin, {"demes_yaml_file": args.yaml})


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args(sys.argv[1:])

    model = build_from_yaml(args.yaml, args.burnin)
    print(model.asblack())
