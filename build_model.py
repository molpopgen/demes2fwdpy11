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

# TODO: we should have an early check on things.
# For example, check that all size change functions are valid, etc..


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
        return int(np.rint(self.converter(input_time)).astype(int))


@attr.s(frozen=True, auto_attribs=True)
class _ModelTimes(object):
    """
    These are in units of the deme graph
    and increase backwards into the past.
    """

    model_start_time: demes.demes.Time
    model_end_time: demes.demes.Time
    model_duration: int = attr.ib(validator=attr.validators.instance_of(int))


@attr.s(auto_attribs=True)
class _MigrationRateChange(object):
    """
    Use to make registry of migration rate changes.

    """

    when: int = attr.ib(
        validator=[demes.demes.non_negative, attr.validators.instance_of(int)]
    )
    source: int = attr.ib(
        validator=[demes.demes.non_negative, attr.validators.instance_of(int)]
    )
    destination: int = attr.ib(
        validator=[demes.demes.non_negative, attr.validators.instance_of(int)]
    )
    rate: float = attr.ib(
        validator=[demes.demes.non_negative, attr.validators.instance_of(float)]
    )
    from_deme_graph: bool = attr.ib(validator=attr.validators.instance_of(bool))


@fwdpy11.class_decorators.attr_class_to_from_dict_no_recurse
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
    migmatrix: typing.Optional[fwdpy11.MigrationMatrix] = None

    # The following do not correspond to fwdpy11 event types.
    migration_rate_changes: typing.List[_MigrationRateChange] = attr.Factory(list)

    def _build_migration_rate_changes(self) -> typing.List[fwdpy11.SetMigrationRates]:
        # TODO: check for setting deme sizes to 0 and set all migration rates
        #       to zero at that time.
        # TODO: test the above for cases where there is migration FROM the deme
        #       whose size got set to zero!!
        set_migration_rates: typing.List[fwdpy11.SetMigrationRates] = []
        return set_migration_rates

    def build_model(self) -> fwdpy11.DiscreteDemography:
        set_migration_rates = self._build_migration_rate_changes()
        return fwdpy11.DiscreteDemography(
            mass_migrations=self.mass_migrations,
            set_deme_sizes=self.set_deme_sizes,
            set_growth_rates=self.set_growth_rates,
            set_selfing_rates=self.set_selfing_rates,
            migmatrix=self.migmatrix,
            set_migration_rates=set_migration_rates,
        )


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


def _get_model_times(dg: demes.DemeGraph) -> _ModelTimes:
    """
    In units of dg.time_units, obtain the following:

    1. The time when the demographic model starts.
    2. The time when it ends.
    3. The total simulation length.
    """
    oldest_deme_time = _get_most_ancient_deme_start_time(dg)
    most_recent_deme_end = _get_most_recent_deme_end_time(dg)

    model_start_time = oldest_deme_time
    if oldest_deme_time == math.inf:
        # Find the end times for all demes
        # with start_time == math.inf.
        # Determine if there's any other deme
        # whose start time is < any of those
        # end times.
        ends_inf = [
            e.end_time for d in dg.demes for e in d.epochs if e.start_time == math.inf
        ]
        starts = [
            e.start_time for d in dg.demes for e in d.epochs if e.start_time != math.inf
        ]
        model_start_time = max([i for i in starts if i >= max(ends_inf)])

    if most_recent_deme_end != 0:
        model_duration = model_start_time - most_recent_deme_end
    else:
        model_duration = model_start_time

    return _ModelTimes(
        model_start_time=model_start_time,
        model_end_time=most_recent_deme_end,
        model_duration=int(np.rint(model_duration)),
    )


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


def _process_epoch(
    deme_id: str,
    e: demes.Epoch,
    idmap: typing.Dict,
    model_times: _ModelTimes,
    burnin_generation: int,
    time_converter: _TimeConverter,
    events: _Fwdpy11Events,
) -> None:
    """
    Can change sizes, cloning rates, and selfing rates.

    Since fwdpy11 currently doesn't understand cloning, we need
    to raise an error if the rate is not None or nonzero.
    """
    when = burnin_generation + time_converter(
        model_times.model_start_time - e.start_time
    )
    if e.selfing_rate is not None:
        if e.start_time == math.inf:
            events.set_selfing_rates.append(
                fwdpy11.SetSelfingRate(when=when, deme=idmap[deme_id], S=e.selfing_rate)
            )

    if e.cloning_rate is not None:
        if e.cloning_rate > 0.0:
            raise ValueError("fwdpy11 does not currently support cloning rates > 0.")

    # Handle size change functions
    if e.start_time != math.inf:
        events.set_deme_sizes.append(
            fwdpy11.SetDemeSize(
                when=when, deme=idmap[deme_id], new_size=e.initial_size,
            )
        )
        if e.final_size != e.initial_size:
            if e.size_function != "exponential":
                raise ValueError(
                    f"Size change function must be exponential.  We got {e.size_function}"
                )
            G = fwdpy11.exponential_growth_rate(
                e.initial_size, e.final_size, int(np.rint(time_converter(e.dt)))
            )
            events.set_growth_rates.append(
                fwdpy11.SetExponentialGrowth(when=when, deme=idmap[deme_id], G=G)
            )
    pass


def _process_all_epochs(
    dg, idmap, model_times, burnin_generation, time_converter, events
):
    for deme in dg.demes:
        for e in deme.epochs:
            _process_epoch(
                deme.id,
                e,
                idmap,
                model_times,
                burnin_generation,
                time_converter,
                events,
            )


def _process_migrations(
    dg: demes.DemeGraph,
    idmap: typing.Dict,
    model_times: _ModelTimes,
    burnin_generation: int,
    time_converter: _TimeConverter,
    events: _Fwdpy11Events,
) -> None:
    """
    Make a record of everything in dg.migrations

    When a migration rate has an end time > 0, it gets entered twice.
    """
    for m in dg.migrations:
        when = burnin_generation + time_converter(
            model_times.model_start_time - m.start_time
        )
        events.migration_rate_changes.append(
            _MigrationRateChange(
                when=when,
                source=idmap[m.source],
                destination=idmap[m.dest],
                rate=m.rate,
                from_deme_graph=True,
            )
        )
        if m.end_time > 0:
            when = burnin_generation + time_converter(
                model_times.model_start_time - m.end_time
            )
            events.migration_rate_changes.append(
                _MigrationRateChange(
                    when=when,
                    source=idmap[m.source],
                    destination=idmap[m.dest],
                    rate=0.0,
                    from_deme_graph=True,
                )
            )


def _process_pulses(
    dg: demes.DemeGraph,
    idmap: typing.Dict,
    model_times: _ModelTimes,
    burnin_generation: int,
    time_converter: _TimeConverter,
    events: _Fwdpy11Events,
) -> None:
    for p in dg.pulses:
        when = burnin_generation + time_converter(model_times.model_start_time - p.time)
        events.migration_rate_changes.append(
            _MigrationRateChange(
                when=when,
                source=idmap[p.source],
                destination=idmap[p.dest],
                rate=p.proportion,
                from_deme_graph=False,
            )
        )


def _process_admixtures(
    dg: demes.DemeGraph,
    idmap: typing.Dict,
    model_times: _ModelTimes,
    burnin_generation: int,
    time_converter: _TimeConverter,
    events: _Fwdpy11Events,
) -> None:
    for a in dg.admixtures:
        when = burnin_generation + time_converter(model_times.model_start_time - a.time)
        for parent, proportion in zip(a.parents, a.proportions):
            events.migration_rate_changes.append(
                _MigrationRateChange(
                    when=when,
                    source=idmap[parent],
                    destination=idmap[a.child],
                    rate=proportion,
                    from_deme_graph=False,
                )
            )


def _process_mergers(
    dg: demes.DemeGraph,
    idmap: typing.Dict,
    model_times: _ModelTimes,
    burnin_generation: int,
    time_converter: _TimeConverter,
    events: _Fwdpy11Events,
) -> None:
    for m in dg.mergers:
        when = burnin_generation + time_converter(model_times.model_start_time - m.time)
        for parent, proportion in zip(m.parents, m.proportions):
            events.migration_rate_changes.append(
                _MigrationRateChange(
                    when=when,
                    source=idmap[parent],
                    destination=idmap[m.child],
                    rate=proportion,
                    from_deme_graph=False,
                )
            )
            # FIXME: the following note isn't right.  We need the migrations
            # to be nonzero at time ``when`` and to be reset the NEXT generation!!
            # NOTE: all migration rates to and from this deme will be set to 0
            # when the final model is built.
            events.set_deme_sizes.append(
                fwdpy11.SetDemeSize(when=when, deme=idmap[parent], new_size=0)
            )


def _process_splits(
    dg: demes.DemeGraph,
    idmap: typing.Dict,
    model_times: _ModelTimes,
    burnin_generation: int,
    time_converter: _TimeConverter,
    events: _Fwdpy11Events,
) -> None:
    """
    A split is a "sudden" creation of > 1 offspring deme
    from a parent and the parent ceases to exist.

    Given that there is no proportions attribute, we infer/assume
    (danger!) that each offspring deme gets 100% of its ancestry
    from the parent.
    """
    for s in dg.splits:
        when = burnin_generation + time_converter(model_times.model_start_time - s.time)
        for c in s.children:
            events.migration_rate_changes.append(
                _MigrationRateChange(
                    when=when,
                    source=idmap[s.parent],
                    destination=idmap[c],
                    rate=1.0,
                    from_deme_graph=False,
                )
            )
        # FIXME: the following note isn't right.  We need the migrations
        # to be nonzero at time ``when`` and to be reset the NEXT generation!!
        # NOTE: all migration rates to and from this deme will be set to 0
        # when the final model is built.
        events.set_deme_sizes.append(
            fwdpy11.SetDemeSize(when=when, deme=idmap[s.parent], new_size=0)
        )


def _process_branches(
    dg: demes.DemeGraph,
    idmap: typing.Dict,
    model_times: _ModelTimes,
    burnin_generation: int,
    time_converter: _TimeConverter,
    events: _Fwdpy11Events,
) -> None:
    """
    A branch creates a child deme with 100% ancestry from the parent.
    The parent continues to exist.

    The 1-to-1 relationship between parent and child means 100% of the
    child's ancestry is from parent.
    """
    for b in dg.branches:
        when = burnin_generation + time_converter(model_times.model_start_time - b.time)
        events.migration_rate_changes.append(
            _MigrationRateChange(
                when=when,
                source=idmap[b.parent],
                destination=idmap[b.child],
                rate=1.0,
                from_deme_graph=False,
            )
        )


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

    burnin_generation = int(np.rint(burnin * Nref))
    model_times = _get_model_times(dg)

    events = _Fwdpy11Events()

    _process_all_epochs(
        dg, idmap, model_times, burnin_generation, time_converter, events
    )
    _process_migrations(
        dg, idmap, model_times, burnin_generation, time_converter, events
    )
    _process_pulses(dg, idmap, model_times, burnin_generation, time_converter, events)
    _process_admixtures(
        dg, idmap, model_times, burnin_generation, time_converter, events
    )
    _process_mergers(dg, idmap, model_times, burnin_generation, time_converter, events)
    _process_splits(dg, idmap, model_times, burnin_generation, time_converter, events)
    _process_branches(dg, idmap, model_times, burnin_generation, time_converter, events)

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
            "burnin_time": burnin_generation,
            "total_simulation_length": burnin_generation
            + time_converter(model_times.model_duration),
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
    print(dg.admixtures)
    print(dg.mergers)
    print(dg.splits)
    print(dg.branches)
    print(dg.pulses)
    return _build_from_deme_graph(dg, burnin, {"demes_yaml_file": args.yaml})


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args(sys.argv[1:])

    model = build_from_yaml(args.yaml, args.burnin)
    print(model.asblack())
