# Cloud ledger, November 2026

Rentals for [X21](X21-phase1-cloud.md) (phase 1 exit cloud checks), approved by the
maintainer on 2026-11-14 with a hard cap of USD 20 gross. Google Cloud, project
the maintainer's Google Cloud project, region us-central1. Spot VMs with a server-side `--max-run-duration=4h`
(termination action DELETE), boot-disk auto-delete and a Mac-side watchdog that deletes by
recorded name and ID.

Prices are the Spot list prices for us-central1 read from Google's Spot pricing page on
2026-11-15; disk is a 50 GB Hyperdisk Balanced at its included performance (about
USD 0.012/h at the October quote, see the remote-compute notes) and the ephemeral IPv4
USD 0.005/h for a Spot VM. Cost = (deleted − created) × (VM + disk + IP), rounded up to the
cent; egress (results only, a few MB) is negligible.

Shared resources of the campaign (free): VPC network `fftn-p1c-net` (9206065056844170441),
subnet `fftn-p1c-sub-usc1` (8650107469094536411), firewall rule `fftn-p1c-ssh`
(3394377833183003859, SSH from the controller's address only).

| Instance | Recorded ID | Type | Zone | Spot USD/h (VM) | Created (UTC) | Deleted (UTC) | Cost USD | Running total |
|---|---|---|---|---:|---|---|---:|---:|
| fftn-p1c-c4d | 4009695476735886453 | c4d-standard-16 | us-central1-a | 0.335 | 2026-11-15 00:19:39 | 2026-11-15 01:30:30 | 0.42 | 0.42 |
| fftn-p1c-c4a | 7016750553389905323 | c4a-standard-16 | us-central1-a | 0.354 | 2026-11-15 01:30:45 | 2026-11-15 01:54:44 | 0.15 | 0.57 |
| fftn-p1c-c4 | 8695178589610787993 | c4-standard-16 | us-central1-a | 0.473 | 2026-11-15 00:19:03 | 2026-11-15 01:54:46 | 0.79 | 1.36 |

Total USD 1.36 estimated gross (cap 20). A first C4D create with `--performance-monitoring-unit`
was refused by the API (not supported for C4D) and created nothing. Teardown verified on
2026-11-15 01:56 UTC: no instance, disk or address left; the campaign's firewall rule, subnet
and network deleted after matching their IDs; the Mac-side watchdogs stopped.

## X31, 2026-11-16

Rentals for [X31](X31-cloud-1d.md) (one-dimensional double on recent server chips),
approved by the maintainer on 2026-11-16 with a cap of USD 20 in total. Same project and
region, Spot VMs with `--max-run-duration=3h` (termination action DELETE), boot-disk
auto-delete, and a Mac-side watchdog per VM that deletes by name only after matching the
recorded ID. Prices as for X21 (Spot list prices read 2026-11-15; the Granite Rapids C4 is
priced as C4), plus 50 GB Hyperdisk Balanced about USD 0.012/h and the ephemeral IPv4
USD 0.005/h. Creation times are those Compute Engine recorded; deletion times are when the
delete call returned.

Shared resources (free): VPC network `fftn-x31-net` (6723768401515729941), subnet
`fftn-x31-sub` (8876375984154564069), firewall rule `fftn-x31-ssh` (4713923710729803259,
SSH from the controller's address only).

| Instance | Recorded ID | Type | Zone | Spot USD/h (VM) | Created (UTC) | Deleted (UTC) | Cost USD | Running total (X31) |
|---|---|---|---|---:|---|---|---:|---:|
| fftn-x31-c4a | 7030097396190631365 | c4a-standard-16 | us-central1-a | 0.354 | 2026-11-16 11:38:19 | 2026-11-16 12:11:37 | 0.21 | 0.21 |
| fftn-x31-c4d | 8950345547308815812 | c4d-standard-16, 1 thread per core | us-central1-a | 0.335 | 2026-11-16 11:38:19 | 2026-11-16 12:12:33 | 0.21 | 0.42 |
| fftn-x31-c4 | 5887323215677052356 | c4-standard-16, 1 thread per core, minimum platform Granite Rapids | us-central1-a | 0.473 | 2026-11-16 11:38:19 | 2026-11-16 12:18:15 | 0.33 | 0.75 |

X31 total USD 0.75 estimated gross (cap 20); the month's total with X21 USD 2.11. Egress
(results only, under 2 MB) is negligible. Teardown verified on 2026-11-16 12:20 UTC: no
instance, disk or address of the campaign left; the firewall rule, subnet and network
deleted after matching their IDs; the watchdogs stopped.

## X31b, 2026-11-16

Rentals for [X31b](X31b-cloud-followup.md) (X32 and X34 on Zen 5 and Granite Rapids), under
the maintainer's 2026-11-16 approval (USD 20 cap for the campaign, at most USD 10 for X31b).
Same project, region, settings and prices as X31: Spot VMs with `--max-run-duration=3h`
(termination action DELETE), boot-disk auto-delete, and a Mac-side watchdog per VM that
deletes by name only after matching the recorded ID. Creation times are those Compute
Engine recorded; deletion times are when the delete call returned.

Shared resources (free): VPC network `fftn-x31b-net` (4896803993719055712), subnet
`fftn-x31b-sub` (4766368096092333426), firewall rule `fftn-x31b-ssh`
(4781569269548098889, SSH from the controller's address only).

| Instance | Recorded ID | Type | Zone | Spot USD/h (VM) | Created (UTC) | Deleted (UTC) | Cost USD | Running total (X31b) |
|---|---|---|---|---:|---|---|---:|---:|
| fftn-x31b-c4d | 1444292870677112145 | c4d-standard-16, 1 thread per core | us-central1-a | 0.335 | 2026-11-16 15:05:34 | 2026-11-16 16:00:00 | 0.32 | 0.32 |
| fftn-x31b-c4 | 7219234991263622447 | c4-standard-16, 1 thread per core, minimum platform Granite Rapids | us-central1-a | 0.473 | 2026-11-16 15:05:38 | 2026-11-16 16:43:28 | 0.80 | 1.12 |

X31b total USD 1.12 estimated gross. With X31 the campaign has spent USD 1.87 of its USD 20
cap; the month's total, with X21, is USD 3.23. Egress (results only, about 15 MB) is
negligible. Teardown was verified on 2026-11-16 16:44 UTC: no instance, disk or address of
the campaign is left; the firewall rule, subnet and network were deleted after matching
their IDs; the watchdogs were stopped.

## X35, 2026-11-16

Rentals for [X35](X35-gnr-profile.md) (a cache-geometry key for Granite Rapids' plans), under
the maintainer's 2026-11-16 approval (USD 20 cap for the campaign, at most USD 6 for X35).
Same project, region, settings and prices as X31b: Spot VMs with `--max-run-duration=3h`
(termination action DELETE), boot-disk auto-delete, and a Mac-side watchdog per VM that
deletes by name only after matching the recorded ID. Creation times are those Compute Engine
recorded; deletion times are when the delete call returned, or Google's preemption.

Shared resources (free): VPC network `fftn-x35-net` (2494351373128600031), subnet
`fftn-x35-sub` (5132173076327729617), firewall rule `fftn-x35-ssh` (8749658591903254953,
SSH from the controller's address only; its source was changed once, when the controller's
address changed).

| Instance | Recorded ID | Type | Zone | Spot USD/h (VM) | Created (UTC) | Deleted (UTC) | Cost USD | Running total (X35) |
|---|---|---|---|---:|---|---|---:|---:|
| fftn-x35-c4 | 4451270156883495296 | c4-standard-16, 1 thread per core, minimum platform Granite Rapids | us-central1-a | 0.473 | 2026-11-16 17:03:43 | 2026-11-16 17:25:47 (preempted by Google) | 0.19 | 0.19 |
| fftn-x35-c4a | 909358881553418641 | c4a-standard-16 | us-central1-a | 0.354 | 2026-11-16 17:03:59 | 2026-11-16 18:09:18 | 0.41 | 0.60 |
| fftn-x35-c4b | 3593494037444099488 | c4-standard-16, 1 thread per core, minimum platform Granite Rapids | us-central1-a | 0.473 | 2026-11-16 17:54:23 | 2026-11-16 20:01:19 | 1.04 | 1.64 |

X35 total USD 1.64 estimated gross. The first Granite Rapids VM was preempted before it
produced a result, so its cost bought nothing. With X31 and X31b the campaign has spent
USD 3.51 of its USD 20 cap; the month's total, with X21, is USD 4.87. Egress (results only,
about 30 MB) is negligible. Teardown was verified on 2026-11-16 at 20:02 UTC: no instance,
disk or address of the campaign is left; the firewall rule, subnet and network were deleted
after matching their IDs; the watchdogs were stopped.

## X45, 2026-11-17

Rentals for [X45](X45-cloud-exit.md) (the phase 1 exit check on Zen 5, Granite Rapids and
Axion), under the maintainer's 2026-11-16 approval (USD 20 cap for the campaign, at most
USD 6 for X45). Same project, region, settings and prices as X35: Spot VMs with
`--max-run-duration=3h` (termination action DELETE), boot-disk auto-delete, and a Mac-side
watchdog per VM that deletes by name only after matching the recorded ID. Creation times are
those Compute Engine recorded; deletion times are when the delete call returned.

Shared resources (free): VPC network `fftn-x45-net` (8456314417235220721), subnet
`fftn-x45-sub` (7877984435035035844), firewall rule `fftn-x45-ssh` (1353138198478444763,
SSH from the controller's address only).

| Instance | Recorded ID | Type | Zone | Spot USD/h (VM) | Created (UTC) | Deleted (UTC) | Cost USD | Running total (X45) |
|---|---|---|---|---:|---|---|---:|---:|
| fftn-x45-c4d | 6028098383780108448 | c4d-standard-16, 1 thread per core | us-central1-a | 0.335 | 2026-11-17 14:36:00 | 2026-11-17 15:01:27 | 0.15 | 0.15 |
| fftn-x45-c4a | 6714390765626316960 | c4a-standard-16 | us-central1-a | 0.354 | 2026-11-17 14:35:59 | 2026-11-17 15:02:37 | 0.17 | 0.32 |
| fftn-x45-c4 | 7614195218821273760 | c4-standard-16, 1 thread per core, minimum platform Granite Rapids | us-central1-a | 0.473 | 2026-11-17 14:35:59 | 2026-11-17 15:06:45 | 0.26 | 0.58 |

X45 total USD 0.58 estimated gross; no VM was preempted. With X31, X31b and X35 the campaign
has spent USD 4.09 of its USD 20 cap; the month's total, with X21, is USD 5.45. Egress
(results only, about 10 MB) is negligible. Teardown was verified on 2026-11-17 at 15:08 UTC:
no instance, disk or address of the campaign is left; the firewall rule, subnet and network
were deleted after matching their IDs; the watchdogs were stopped.

## X48, 2026-11-18

Rentals for [X48](X48-exit-3.md) (the phase 1 exit check on Zen 5, Granite Rapids and Axion
again, `main` 8c18dd2), approved by the maintainer on 2026-11-17 night (USD 20 cap for the
campaign, at most USD 6 for X48). Same project, region, settings and prices as X45, except
`--max-run-duration=2h` (termination action DELETE); boot-disk auto-delete, and a Mac-side
watchdog per VM (2 h 15 min) that deletes by name only after matching the recorded ID.
Creation times are those Compute Engine recorded; deletion times are when the delete call
returned.

Shared resources (free): VPC network `fftn-x48-net` (5903739121847849715), subnet
`fftn-x48-sub` (3524167475670373060), firewall rule `fftn-x48-ssh` (4522014242547782359,
SSH from the controller's address only).

| Instance | Recorded ID | Type | Zone | Spot USD/h (VM) | Created (UTC) | Deleted (UTC) | Cost USD | Running total (X48) |
|---|---|---|---|---:|---|---|---:|---:|
| fftn-x48-c4a | 5987786258826482065 | c4a-standard-16 | us-central1-a | 0.354 | 2026-11-18 00:21:18 | 2026-11-18 00:47:47 | 0.17 | 0.17 |
| fftn-x48-c4d | 9061261565899133329 | c4d-standard-16, 1 thread per core | us-central1-a | 0.335 | 2026-11-18 00:21:19 | 2026-11-18 00:49:01 | 0.17 | 0.34 |
| fftn-x48-c4 | 6931765661841576336 | c4-standard-16, 1 thread per core, minimum platform Granite Rapids | us-central1-a | 0.473 | 2026-11-18 00:21:20 | 2026-11-18 00:56:01 | 0.29 | 0.63 |

X48 total USD 0.63 estimated gross; no VM was preempted. With X31, X31b, X35 and X45 the
campaign has spent USD 4.72 of its USD 20 cap; the month's total, with X21, is USD 6.08.
Egress (results only, about 10 MB) is negligible. Teardown was verified on 2026-11-18 at
00:57 UTC: no instance, disk or address of the campaign is left; the firewall rule, subnet
and network were deleted after matching their IDs; the watchdogs were stopped.
