import argparse
import math
import sys
import typing
import warnings
import copy

import attr
import demes
import fwdpy11
import fwdpy11.class_decorators
import fwdpy11.demographic_models
import numpy as np

# TODO: we should have an early check on things.
# For example, check that all size change functions are valid, etc..

# TODO: we need to have a test case for a final size of zero.


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
    rate_change: float = attr.ib(validator=[attr.validators.instance_of(float)])
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
    idmap: typing.Dict = None

    # The initial continuous migration matrix
    initial_migmatrix: typing.Optional[fwdpy11.MigrationMatrix] = None
    # The migration matrix that we update to get changes in migration rates
    migmatrix: typing.Optional[fwdpy11.MigrationMatrix] = None

    # The following do not correspond to fwdpy11 event types.
    migration_rate_changes: typing.List[_MigrationRateChange] = attr.Factory(list)
    # deme_extinctions: typing.List[_DemeExtinctionEvent] = attr.Factory(list)

    def _update_changes_at_m(self, changes_at_m, migration_rate_change):
        # tally changes
        if migration_rate_change.from_deme_graph:
            changes_at_m[0][migration_rate_change.destination][
                migration_rate_change.source
            ] += migration_rate_change.rate_change
            if migration_rate_change.destination != migration_rate_change.source:
                changes_at_m[0][migration_rate_change.destination][
                    migration_rate_change.destination
                ] -= migration_rate_change.rate_change
        else:
            changes_at_m[1][migration_rate_change.destination][
                migration_rate_change.source
            ] += migration_rate_change.rate_change
            changes_at_m[1][migration_rate_change.destination][
                migration_rate_change.destination
            ] -= migration_rate_change.rate_change

    def _update_continuous_mass_migrations(self, changes_at_m, M_cont, M_mass):
        M_cont += changes_at_m[0]
        M_mass += changes_at_m[1]
        return M_cont, M_mass

    def _migration_matrix_from_partition(self, M_cont, M_mass):
        new_migmatrix = np.diag(np.diag(M_mass)).dot(M_cont) + (
            M_mass - np.diag(np.diag(M_mass))
        )
        return new_migmatrix

    def _build_migration_rate_changes(self) -> typing.List[fwdpy11.SetMigrationRates]:
        # We track the coninuous migration rates, and then augment with a matrix that
        # specifies changes to migration due to "instantaneous" events. The
        # instantaneous migration matrix is typically just the identity matrix,
        # (i.e. uses continuous rates, but off diag elements scale continuous rates
        # and add to ancestry source from the off diagonal source column)
        # but for some generations has ancestry pointing to different demes due
        # to pulse, split, etc events
        if self.migmatrix is not None:
            M_cont = copy.deepcopy(self.migmatrix)
        else:
            M_cont = np.eye(len(self.idmap))
        M_mass = np.eye(len(self.idmap))

        set_migration_rates: typing.List[fwdpy11.SetMigrationRates] = []

        self.migration_rate_changes = sorted(
            self.migration_rate_changes,
            key=lambda x: (x.when, x.destination, x.source, x.from_deme_graph),
        )
        # self.deme_extinctions = sorted(
        #    self.deme_extinctions, key=lambda x: (x.when, x.deme)
        # )
        m = 0
        changes_at_m = [
            np.zeros((len(self.idmap), len(self.idmap))),  # from DemeGraph
            np.zeros((len(self.idmap), len(self.idmap))),  # not from DemeGraph
        ]
        while m < len(self.migration_rate_changes):
            # gather all migration rate changes and extinction events
            # that occur at a given time
            self._update_changes_at_m(changes_at_m, self.migration_rate_changes[m])
            mm = m + 1

            while (
                mm < len(self.migration_rate_changes)
                and self.migration_rate_changes[mm].when
                == self.migration_rate_changes[m].when
            ):
                self._update_changes_at_m(changes_at_m, self.migration_rate_changes[mm])
                mm += 1

            # update M_cont and M_mass
            M_cont, M_mass = self._update_continuous_mass_migrations(
                changes_at_m, M_cont, M_mass
            )
            # get the new migration matrix
            new_migmatrix = self._migration_matrix_from_partition(M_cont, M_mass)
            # for any rows that don't match, add a fwdpy11.SetMigrationRate
            for i in range(len(self.idmap)):
                if np.any(self.migmatrix[i] != new_migmatrix[i]):
                    set_migration_rates.append(
                        fwdpy11.SetMigrationRates(
                            self.migration_rate_changes[m].when, i, new_migmatrix[i]
                        )
                    )

            self.migmatrix = new_migmatrix

            m = mm
            # reset changes
            changes_at_m = [
                np.zeros((len(self.idmap), len(self.idmap))),  # from DemeGraph
                np.zeros((len(self.idmap), len(self.idmap))),  # not from DemeGraph
            ]

        return set_migration_rates

    def build_model(self) -> fwdpy11.DiscreteDemography:
        set_migration_rates = self._build_migration_rate_changes()
        return fwdpy11.DiscreteDemography(
            mass_migrations=self.mass_migrations,
            set_deme_sizes=self.set_deme_sizes,
            set_growth_rates=self.set_growth_rates,
            set_selfing_rates=self.set_selfing_rates,
            migmatrix=self.initial_migmatrix,
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
    return max([d.start_time for d in dg.demes])


def _get_most_recent_deme_end_time(dg: demes.DemeGraph) -> demes.demes.Time:
    return min([d.end_time for d in dg.demes])


def _get_model_times(dg: demes.DemeGraph) -> _ModelTimes:
    """
    In units of dg.time_units, obtain the following:

    1. The time when the demographic model starts.
    2. The time when it ends.
    3. The total simulation length.

    """
    # FIXME: this function isn't working well.
    # For example, twodemes.yml and twodemes_one_goes_away.yml
    # both break it.
    oldest_deme_time = _get_most_ancient_deme_start_time(dg)
    most_recent_deme_end = _get_most_recent_deme_end_time(dg)

    model_start_time = oldest_deme_time
    if oldest_deme_time == math.inf:
        # We want to find the time of first event or
        # the first demographic change, which is when
        # burnin will end. To do this, get a list of
        # first size change for all demes with inf
        # start time, and the start time for all other
        # demes, and take max of those.
        ends_inf = [d.epochs[0].end_time for d in dg.demes if d.start_time == math.inf]
        starts = [d.start_time for d in dg.demes if d.start_time != math.inf]
        mig_starts = [m.start_time for m in dg.migrations if m.start_time != math.inf]
        mig_ends = [m.end_time for m in dg.migrations if m.start_time == math.inf]
        pulse_times = [p.time for p in dg.pulses]
        model_start_time = max(ends_inf + starts + mig_starts + mig_ends + pulse_times)

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
    Set any migration rates that have start time of inf. More
    recent migration rate changes or start time set the model
    start time at or before that time
    """
    if len(idmap) > 1:
        migmatrix = np.zeros((len(idmap), len(idmap)))
        for deme_id, ii in idmap.items():
            if dg[deme_id].start_time == math.inf:
                migmatrix[ii, ii] = 1.0
        if len(dg.migrations) > 0:
            for m in dg.migrations:
                if m.start_time == math.inf:
                    migmatrix[idmap[m.dest]][idmap[m.source]] += m.rate
                    migmatrix[idmap[m.dest]][idmap[m.dest]] -= m.rate

        events.migmatrix = migmatrix
        events.initial_migmatrix = migmatrix


def _process_epoch(
    deme_id: str,
    e: demes.Epoch,
    idmap: typing.Dict,
    model_times: _ModelTimes,
    burnin_generation: int,
    events: _Fwdpy11Events,
) -> None:
    """
    Can change sizes, cloning rates, and selfing rates.

    Since fwdpy11 currently doesn't understand cloning, we need
    to raise an error if the rate is not None or nonzero.
    """
    if e.start_time != math.inf:
        when = burnin_generation + int(model_times.model_start_time - e.start_time)
    else:
        when = 0

    if e.selfing_rate is not None:
        events.set_selfing_rates.append(
            fwdpy11.SetSelfingRate(when=when, deme=idmap[deme_id], S=e.selfing_rate)
        )

    if e.cloning_rate is not None:
        if e.cloning_rate > 0.0:
            raise ValueError("fwdpy11 does not currently support cloning rates > 0.")

    if e.start_time != math.inf:
        # Handle size change functions
        events.set_deme_sizes.append(
            fwdpy11.SetDemeSize(
                when=when - 1,
                deme=idmap[deme_id],
                new_size=e.initial_size,
            )
        )
        if e.final_size != e.initial_size:
            if e.size_function != "exponential":
                raise ValueError(
                    f"Size change function must be exponential.  We got {e.size_function}"
                )
            G = fwdpy11.exponential_growth_rate(
                e.initial_size, e.final_size, int(np.rint(e.time_span))
            )
            events.set_growth_rates.append(
                fwdpy11.SetExponentialGrowth(when=when, deme=idmap[deme_id], G=G)
            )


def _process_all_epochs(dg, idmap, model_times, burnin_generation, events):
    """
    Processes all epochs of all demes to set sizes and selfing rates.
    """
    for deme in dg.demes:
        for e in deme.epochs:
            _process_epoch(
                deme.id,
                e,
                idmap,
                model_times,
                burnin_generation,
                events,
            )
        # if a deme starts more recently than math.inf, we have to
        # turn on migration in that deme with a diagonal element to 1
        if deme.start_time < math.inf:
            events.migration_rate_changes.append(
                _MigrationRateChange(
                    when=burnin_generation
                    + int(model_times.model_start_time - deme.start_time),
                    source=idmap[deme.id],
                    destination=idmap[deme.id],
                    rate_change=1.,
                    from_deme_graph=True,
                )
            )

        # if a deme ends before time zero, we set diag entry in migmatrix to 0
        if deme.end_time > 0:
            events.migration_rate_changes.append(
                _MigrationRateChange(
                    when=burnin_generation
                    + int(model_times.model_start_time - deme.end_time),
                    source=idmap[deme.id],
                    destination=idmap[deme.id],
                    rate_change=-1.,
                    from_deme_graph=True,
                )
            )

        # if deme ends before time zero, we set set its size to zero
        # we proces deme extintions here instead of in the events
        if deme.end_time > 0:
            events.set_deme_sizes.append(
                fwdpy11.SetDemeSize(
                    when=burnin_generation
                    + int(model_times.model_start_time - deme.end_time),
                    deme=idmap[deme.id],
                    new_size=0,
                )
            )


def _process_migrations(
    dg: demes.DemeGraph,
    idmap: typing.Dict,
    model_times: _ModelTimes,
    burnin_generation: int,
    events: _Fwdpy11Events,
) -> None:
    """
    Make a record of everything in dg.migrations

    When a migration rate has an end time > 0, it gets entered twice.
    """
    for m in dg.migrations:
        if m.start_time < math.inf:
            when = burnin_generation + int(model_times.model_start_time - m.start_time)
            events.migration_rate_changes.append(
                _MigrationRateChange(
                    when=when,
                    source=idmap[m.source],
                    destination=idmap[m.dest],
                    rate_change=m.rate,
                    from_deme_graph=True,
                )
            )
        if m.end_time > 0:
            when = burnin_generation + int(model_times.model_start_time - m.end_time)
            events.migration_rate_changes.append(
                _MigrationRateChange(
                    when=when,
                    source=idmap[m.source],
                    destination=idmap[m.dest],
                    rate_change=-m.rate,
                    from_deme_graph=True,
                )
            )


def _process_pulses(
    dg: demes.DemeGraph,
    idmap: typing.Dict,
    model_times: _ModelTimes,
    burnin_generation: int,
    events: _Fwdpy11Events,
) -> None:
    ### we want to change this to play nicely with continuous migration: the
    ### 1-p.proportion of ancestry still behaves according to the continuous
    ### migration that may be going on.
    ### FIX ME
    for p in dg.pulses:
        when = burnin_generation + int(model_times.model_start_time - p.time)
        events.migration_rate_changes.append(
            _MigrationRateChange(
                when=when - 1,
                source=idmap[p.source],
                destination=idmap[p.dest],
                rate_change=p.proportion,
                from_deme_graph=False,
            )
        )
        events.migration_rate_changes.append(
            _MigrationRateChange(
                when=when,
                source=idmap[p.source],
                destination=idmap[p.dest],
                rate_change=-p.proportion,
                from_deme_graph=False,
            )
        )


def _process_admixtures(
    dg: demes.DemeGraph,
    idmap: typing.Dict,
    model_times: _ModelTimes,
    burnin_generation: int,
    events: _Fwdpy11Events,
) -> None:
    for a in dg.admixtures:
        when = burnin_generation + int(model_times.model_start_time - a.time)
        for parent, proportion in zip(a.parents, a.proportions):
            events.migration_rate_changes.append(
                _MigrationRateChange(
                    when=when - 1,
                    source=idmap[parent],
                    destination=idmap[a.child],
                    rate_change=proportion,
                    from_deme_graph=False,
                )
            )
            events.migration_rate_changes.append(
                _MigrationRateChange(
                    when=when,
                    source=idmap[parent],
                    destination=idmap[a.child],
                    rate_change=-proportion,
                    from_deme_graph=False,
                )
            )


def _process_mergers(
    dg: demes.DemeGraph,
    idmap: typing.Dict,
    model_times: _ModelTimes,
    burnin_generation: int,
    events: _Fwdpy11Events,
) -> None:
    for m in dg.mergers:
        when = burnin_generation + int(model_times.model_start_time - m.time)
        for parent, proportion in zip(m.parents, m.proportions):
            events.migration_rate_changes.append(
                _MigrationRateChange(
                    when=when - 1,
                    source=idmap[parent],
                    destination=idmap[m.child],
                    rate_change=proportion,
                    from_deme_graph=False,
                )
            )
            events.migration_rate_changes.append(
                _MigrationRateChange(
                    when=when,
                    source=idmap[parent],
                    destination=idmap[m.child],
                    rate_change=-proportion,
                    from_deme_graph=False,
                )
            )


def _process_splits(
    dg: demes.DemeGraph,
    idmap: typing.Dict,
    model_times: _ModelTimes,
    burnin_generation: int,
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
        when = burnin_generation + int(model_times.model_start_time - s.time)
        for c in s.children:
            # one generation of migration to move lineages from parent to children
            events.migration_rate_changes.append(
                _MigrationRateChange(
                    when=when - 1,
                    source=idmap[s.parent],
                    destination=idmap[c],
                    rate_change=1.0,
                    from_deme_graph=False,
                )
            )
            # turn off that migration after one generation
            events.migration_rate_changes.append(
                _MigrationRateChange(
                    when=when,
                    source=idmap[s.parent],
                    destination=idmap[c],
                    rate_change=-1.0,
                    from_deme_graph=False,
                )
            )


def _process_branches(
    dg: demes.DemeGraph,
    idmap: typing.Dict,
    model_times: _ModelTimes,
    burnin_generation: int,
    events: _Fwdpy11Events,
) -> None:
    """
    A branch creates a child deme with 100% ancestry from the parent.
    The parent continues to exist.

    The 1-to-1 relationship between parent and child means 100% of the
    child's ancestry is from parent.
    """
    for b in dg.branches:
        when = burnin_generation + int(model_times.model_start_time - b.time)
        # turn on migration for one generation at "when"
        events.migration_rate_changes.append(
            _MigrationRateChange(
                when=when - 1,
                source=idmap[b.parent],
                destination=idmap[b.child],
                rate_change=1.0,
                from_deme_graph=False,
            )
        )
        # end that migration after one generation
        events.migration_rate_changes.append(
            _MigrationRateChange(
                when=when,
                source=idmap[b.parent],
                destination=idmap[b.child],
                rate_change=-1.0,
                from_deme_graph=False,
            )
        )


def build_from_deme_graph(
    dg: demes.DemeGraph, burnin: int, source: typing.Optional[typing.Dict] = None
) -> fwdpy11.demographic_models.DemographicModelDetails:
    """
    The workhorse.
    """
    # dg must be in generations - replaces time_converter
    dg = dg.in_generations()

    idmap = _build_deme_id_to_int_map(dg)
    initial_sizes = _get_initial_deme_sizes(dg, idmap)
    Nref = _get_ancestral_population_size(dg)

    burnin_generation = int(np.rint(burnin * Nref))
    model_times = _get_model_times(dg)

    events = _Fwdpy11Events(idmap=idmap)

    _set_initial_migration_matrix(dg, idmap, events)
    _process_all_epochs(dg, idmap, model_times, burnin_generation, events)
    _process_migrations(dg, idmap, model_times, burnin_generation, events)
    _process_pulses(dg, idmap, model_times, burnin_generation, events)
    _process_admixtures(dg, idmap, model_times, burnin_generation, events)
    _process_mergers(dg, idmap, model_times, burnin_generation, events)
    _process_splits(dg, idmap, model_times, burnin_generation, events)
    _process_branches(dg, idmap, model_times, burnin_generation, events)

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
            "total_simulation_length": burnin_generation + model_times.model_duration,
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
    return build_from_deme_graph(dg, burnin, {"demes_yaml_file": filename})


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args(sys.argv[1:])

    model = build_from_yaml(args.yaml, args.burnin)
    print(model.asblack())
