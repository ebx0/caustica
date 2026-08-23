# Changelog

Nothing has been released yet: `0.1.0` is the first planned release, and it is gated on the
ITRUSST benchmark suite — all nine cases, acoustic-only — rather than on a date. What follows
is what is on `master` today.

This page **is** the repository's `CHANGELOG.md`, included rather than copied, so the two
cannot drift apart. The milestone ledger behind it — including the rule that no box is ticked
without measured evidence — is
[MILESTONES.md](https://github.com/ebx0/caustica/blob/master/MILESTONES.md), and the long-form
account of why each decision went the way it did is the [engineering log](devlog.md).

!!! info "Nothing below is a stability promise"

    Between milestones the API moves. Four surfaces are meant to be depended on — the job
    schema, the documented [Python API](api/index.md), the
    [five extension points](extending.md) and the [GUI contract](gui_contract.md) — and even
    those carry a `/1` in their names so a break can be announced rather than discovered.

---

--8<-- "CHANGELOG.md"
