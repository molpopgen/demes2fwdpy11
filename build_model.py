import argparse
import sys
import typing

import attr
import demes
import fwdpy11
import fwdpy11.class_decorators
import fwdpy11.demographic_models


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


def _get_most_ancient_deme_time(dg: demes.DemeGraph) -> typing.Optional[int]:
    oldest_deme_time = None
    for deme in dg.demes:
        if oldest_deme_time is None:
            oldest_deme_time = deme.epochs[0].start_time
        else:
            oldest_deme_time = max(oldest_deme_time, deme.epochs[0].start_time)
    if oldest_deme_time is None:
        raise ValueError(f"invalid most ancient deme start_time: {oldest_deme_time}")
    return oldest_deme_time


def _get_ancestral_population_size(dg: demes.DemeGraph) -> int:
    """
    Need this for the burnin time.

    If there are > 1 demes with the same most ancient start_time,
    then the ancestral size is considered to be the size
    of all those demes (size of ancestral metapopulation).
    """
    oldest_deme_time = _get_most_ancient_deme_time(dg)

    Nref = 0
    for deme in dg.demes:
        if deme.start_time == oldest_deme_time:
            if Nref is None:
                Nref = deme.epochs[0].initial_size
            else:
                Nref += deme.epochs[0].initial_size
    if Nref <= 0:
        raise ValueError(f"invalid ancestral population size: {Nref}")
    return Nref


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


def _process_epoch(e: demes.Epoch, idmap: typing.Dict, events: _Fwdpy11Events) -> None:
    pass


def _process_migration(
    m: demes.Migration, idmap: typing.Dict, events: _Fwdpy11Events
) -> None:
    pass


def _process_pulse(p: demes.Pulse, idmap: typing.Dict, events: _Fwdpy11Events) -> None:
    pass


def _build_from_deme_graph(
    input_model: demes.DemeGraph, burnin: int
) -> fwdpy11.demographic_models.DemographicModelDetails:
    """
    The workhorse.
    """
    idmap = _build_deme_id_to_int_map(input_model)
    Nref = _get_ancestral_population_size(input_model)
    events = _Fwdpy11Events()

    doi = None
    if input_model.doi != "None":
        doi = input_model.doi
    return fwdpy11.demographic_models.DemographicModelDetails(
        model=events.build_model(),
        name=input_model.description,
        source={"function": "_build_from_deme_graph"},
        parameters=None,
        citation=fwdpy11.demographic_models.DemographicModelCitation(
            DOI=doi, full_citation=None, metadata=None
        ),
        metadata={j: i for i, j in idmap.items()},
    )


def build_from_yaml(
    filename: typing.Text, burnin: int
) -> fwdpy11.demographic_models.DemographicModelDetails:
    """
    Candidate for user-facing function.
    Could also be a part of the DiscreteDemography
    public interface, although static functions are odd in Python?
    """
    input_model = demes.load(filename)
    return _build_from_deme_graph(input_model, burnin)


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args(sys.argv[1:])

    model = build_from_yaml(args.yaml, args.burnin)
    print(model.asblack())
