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

## X55, 2026-11-18

Rental for [X55](X55-gnr-depth.md) (the depth-first chain on Granite Rapids), approved by
the maintainer on 2026-11-18 ("yes, run Granite Rapids within the cap"; at most USD 2).
Same project, region, prices and safeguards as X48: `--max-run-duration=150m` with
termination action DELETE, boot-disk auto-delete, a guest shutdown after 145 minutes and a
Mac-side watchdog (155 minutes) deleting by name only after matching the recorded ID.

Shared resources (free): VPC network `fftn-x55-net` (7806003257055173776), subnet
`fftn-x55-sub` (5101221257069780067), firewall rule `fftn-x55-ssh` (582455747989789818,
SSH from the controller's address only).

| Instance | Recorded ID | Type | Zone | Spot USD/h (VM) | Created (UTC) | Deleted (UTC) | Cost USD | Running total (X55) |
|---|---|---|---|---:|---|---|---:|---:|
| fftn-x55-c4 | 2451080659952786504 | c4-standard-16, 1 thread per core, minimum platform Granite Rapids | us-central1-a | 0.473 | 2026-11-18 08:32:39 | 2026-11-18 09:23:30 | 0.40 | 0.40 |

X55 total USD 0.40 estimated gross; not preempted. The campaign has spent USD 5.12 of its
USD 20 cap; the month's total, with X21, is USD 6.48. Teardown verified on 2026-11-18 at
09:24 UTC: the instance was deleted after matching its ID, then the firewall rule, subnet
and network after matching theirs; the inventory lists no instance, no `fftn` disk or
firewall rule and only the `default` network; the watchdog was stopped.

## X56 and X58, 2026-11-18 and 19

Rentals for [X56](X56-exit-4.md) (the phase 1 exit check on the cloud cores: Zen 5, Axion,
Granite Rapids, and two Intel families the rules reach unmeasured, Sapphire Rapids and Ice Lake)
and [X58](X58-intel-keys.md) (the profile keys that check called for), all in us-central1-a,
Spot, created by the lead session under the maintainer's cap (USD 20 for the campaign).
Safeguards as in X55: `--max-run-duration=150m` with termination action DELETE, boot-disk
auto-delete, a guest shutdown after 145 minutes, and a Mac-side watchdog (150–165 minutes) that
deleted by name only after matching the recorded ID. Spot prices for C3 and N2 are from the
Cloud Billing catalogue on 2026-11-18 (vCPU and memory rates, billed for all vCPUs with one
thread per core); the others as before. A VM marked preempted ends where its loss was first
seen (an upper bound). Shared resources (free): network `fftn-x56-net` (1594255085495402557),
subnet `fftn-x56-sub` (6565651907817259020), firewall rule `fftn-x56-ssh` (7001961555060342785,
SSH from the controller's address only).

| Instance | Recorded ID | Type | Spot USD/h | Created (UTC) | Deleted or lost (UTC) | Cost USD | Running total | Use |
|---|---|---|---:|---|---|---:|---:|---|
| fftn-x56-c3 | 4802825887329304592 | c3-standard-8, 1 thread per core | 0.150 | 2026-11-18 19:39:44 | 2026-11-18 21:12:10 | 0.23 | 0.23 | exit check, padding (Sapphire Rapids) |
| fftn-x56-c4a | 2341279391533776912 | c4a-standard-16 | 0.354 | 2026-11-18 19:39:44 | 2026-11-18 20:45:06 | 0.39 | 0.62 | exit check (Axion) |
| fftn-x56-n2 | 2367672626992660496 | n2-standard-16, 1 thread per core | 0.443 | 2026-11-18 19:39:43 | 2026-11-18 21:17:00 | 0.72 | 1.34 | exit check, padding (Ice Lake) |
| fftn-x56-c4 | 344820189664133136 | c4-standard-16, 1 thread per core | 0.473 | 2026-11-18 19:39:43 | 2026-11-18 20:49:18 | 0.55 | 1.88 | exit check, padding (Granite Rapids) |
| fftn-x56-c4d | 5160643053027868688 | c4d-standard-16, 1 thread per core | 0.335 | 2026-11-18 19:39:44 | 2026-11-18 20:33:45 | 0.30 | 2.19 | Zen 5, preempted, no results |
| fftn-x56-c4d | 8060973165948678284 | c4d-standard-16, 1 thread per core | 0.335 | 2026-11-18 20:45:23 | 2026-11-18 21:44:13 | 0.33 | 2.51 | exit check, padding (Zen 5) |
| fftn-x56-n2b | 1834219031491750143 | n2-standard-16, 1 thread per core | 0.443 | 2026-11-18 21:18:08 | 2026-11-18 22:26:23 | 0.50 | 3.02 | X58 screens (Ice Lake) |
| fftn-x56-c3b | 6555559202823317759 | c3-standard-8, 1 thread per core | 0.150 | 2026-11-18 21:18:09 | 2026-11-18 22:10:11 | 0.13 | 3.15 | X58 screen (Sapphire Rapids) |
| fftn-x56-c4b | 5676626624239610110 | c4-standard-16, 1 thread per core | 0.473 | 2026-11-18 21:18:09 | 2026-11-18 21:44:34 | 0.21 | 3.36 | preempted, no results |
| fftn-x56-c4c | 5137926575862552518 | c4-standard-16, 1 thread per core | 0.473 | 2026-11-18 22:39:37 | 2026-11-18 23:04:20 | 0.19 | 3.55 | preempted, no results |
| fftn-x56-n2c | 3501819253295912902 | n2-standard-16, 1 thread per core | 0.443 | 2026-11-18 22:39:37 | 2026-11-19 00:39:42 | 0.89 | 4.44 | X58 judging (Ice Lake) |
| fftn-x56-c3c | 8955151675842336710 | c3-standard-8, 1 thread per core | 0.150 | 2026-11-18 22:39:37 | 2026-11-19 00:12:58 | 0.23 | 4.67 | preempted, no results |
| fftn-x56-c3e | 6094541188546952388 | c3-standard-8, 1 thread per core | 0.150 | 2026-11-19 00:26:19 | 2026-11-19 02:09:40 | 0.26 | 4.93 | X58 judging (Sapphire Rapids) |
| fftn-x56-c4e | 1275711212420857028 | c4-standard-16, 1 thread per core | 0.473 | 2026-11-19 00:26:19 | 2026-11-19 01:39:53 | 0.58 | 5.51 | X58 judging (Granite Rapids) |
| fftn-x56-c4f | 3048537879573947694 | c4-standard-16, 1 thread per core | 0.473 | 2026-11-19 02:15:29 | 2026-11-19 03:16:08 | 0.48 | 5.99 | final exit check (Granite Rapids) |
| fftn-x56-n2f | 450551761822016813 | n2-standard-16, 1 thread per core | 0.443 | 2026-11-19 02:15:30 | 2026-11-19 04:04:34 | 0.81 | 6.79 | final exit check (Ice Lake) |
| fftn-x56-c3f | 7382853677370041646 | c3-standard-8, 1 thread per core | 0.150 | 2026-11-19 02:15:30 | 2026-11-19 04:07:36 | 0.28 | 7.07 | final exit check (Sapphire Rapids) |

X56 and X58 total USD 7.07 estimated gross (four VMs preempted, three of them without
results). With the earlier experiments the campaign has spent USD 12.19 of its USD 20
cap. Teardown verified on 2026-11-19 at 04:15 UTC: every instance above deleted (by the job
loop after matching its ID, by its termination action, or on preemption), then the firewall
rule, subnet and network after matching their IDs; the inventory lists no `fftn` instance, disk
or firewall rule and only the `default` network; every watchdog stopped.

## X64 and the client catalogue check, 2026-11-24

Rentals for [X64](X64-cloud-screen-1124.md) (the reach screens and X50 on recent Intel) and
for the AArch64 leg of the client catalogue check (one Axion VM). The maintainer approved
USD 5 for these, within the campaign's USD 20. Same project and region, zone us-central1-a,
Spot VMs with `--max-run-duration=150m` (termination action DELETE), boot-disk auto-delete,
and a Mac-side watchdog per VM that deletes by name only after matching the recorded ID.
Prices as before (Spot list prices read 2026-11-15). The n2d and c3d prices were not on
that list and are estimates. Add about USD 0.017/h per VM for disk and IP. Creation times
are those Compute Engine recorded. Deletion times are when the delete call returned.
Shared resources (free): network `fftn-s1124-net` (155638627033470172), subnet
`fftn-s1124-sub` (5706820330252194990), firewall rule `fftn-s1124-ssh`
(6255819946428257445, SSH from the controller's address only).

| Instance | Recorded ID | Type | Spot USD/h | Created (UTC) | Deleted (UTC) | Cost USD | Running total | Use |
|---|---|---|---:|---|---|---:|---:|---|
| fftn-s1124-n2d | 1373423097111696560 | n2d-standard-16, 1 thread per core | 0.400 (est.) | 2026-11-24 10:43:44 | 2026-11-24 11:33:16 | 0.35 | 0.35 | Zen 2 reach screens (X57, X61, X62, X63) |
| fftn-s1124-c3d | 5993219272219350158 | c3d-standard-16, 1 thread per core | 0.330 (est.) | 2026-11-24 10:43:45 | 2026-11-24 11:34:15 | 0.30 | 0.65 | X63 reach screen (Zen 4) |
| fftn-s1124-c4e | 2250537732003124366 | c4-standard-16, 1 thread per core | 0.473 | 2026-11-24 10:43:46 | 2026-11-24 11:20:22 | 0.30 | 0.95 | X58 and X50 reach, exit check (Emerald Rapids) |
| fftn-s1124-c3 | 8964760711706760334 | c3-standard-8, 1 thread per core | 0.150 | 2026-11-24 10:43:47 | 2026-11-24 11:14:03 | 0.09 | 1.04 | X50 and diagnosis (Sapphire Rapids) |
| fftn-s1124-n2 | 6700702670516704398 | n2-standard-16, 1 thread per core | 0.443 | 2026-11-24 10:43:48 | 2026-11-24 11:35:14 | 0.40 | 1.44 | X63 reach screen (Ice Lake) |
| fftn-s1124-c4 | 2202191046088642026 | c4-standard-16, 1 thread per core | 0.473 | 2026-11-24 11:20:37 | 2026-11-24 12:18:14 | 0.48 | 1.92 | Granite Rapids: package install hung, no results |
| fftn-s1124-c4a | 3936476630937551868 | c4a-standard-16 | 0.354 | 2026-11-24 12:03:31 | 2026-11-24 13:03:16 | 0.37 | 2.29 | client catalogue check (Axion) |
| fftn-s1124-c4b | 3929449055123947132 | c4-standard-16, 1 thread per core | 0.473 | 2026-11-24 12:18:28 | 2026-11-24 12:36:11 | 0.15 | 2.44 | X50 and diagnosis (Granite Rapids) |

The total is USD 2.44, an estimated gross figure. Granite Rapids' first VM was created only
after the Emerald VM had been deleted, because the region's C4 quota is 24 vCPUs. Its
package install then hung for 55 minutes on a mirror, so it was deleted by ID without
results, and its replacement installed with retries. With the earlier experiments the
campaign has spent USD 14.63 of its USD 20 cap. Teardown was verified on 2026-11-24 at
13:04 UTC:
- every instance above was deleted after its ID was matched;
- the firewall rule, subnet and network were then deleted after their IDs were matched;
- the project lists only the `default` network and its rules;
- every watchdog was stopped.

## X69, 2026-11-24

Rentals for [X69](X69-intel-radices.md) (two-pass rows of 1024, 160 and 240 on recent Intel). The
maintainer approved USD 3. The run used the same project, region and zone, Spot VMs with
`--max-run-duration=150m` (termination action DELETE), boot-disk auto-delete and a Mac-side
watchdog per VM, prices as before plus about USD 0.017/h for disk and IP. Emerald Rapids was
created after Granite Rapids was deleted (the region's C4 quota is 24 vCPUs). Shared resources
(free): network `fftn-x69-net` (2186203687459227221), subnet `fftn-x69-sub`
(1122734089496430117), firewall rule `fftn-x69-ssh` (4633249729418364476, SSH from the
controller's address only).

| Instance | Recorded ID | Type | Spot USD/h | Created (UTC) | Deleted (UTC) | Cost USD | Running total | Use |
|---|---|---|---:|---|---|---:|---:|---|
| fftn-x69-c4 | 1802874914238629383 | c4-standard-16, 1 thread per core | 0.473 | 2026-11-24 16:18:48 | 2026-11-24 16:36:43 | 0.15 | 0.15 | X69 judging (Granite Rapids) |
| fftn-x69-c3 | 8859902903400793622 | c3-standard-8, 1 thread per core | 0.150 | 2026-11-24 16:19:05 | 2026-11-24 16:46:40 | 0.08 | 0.23 | X69 judging (Sapphire Rapids) |
| fftn-x69-c4e | 3321296984683840497 | c4-standard-16, 1 thread per core | 0.473 | 2026-11-24 16:36:47 | 2026-11-24 16:54:18 | 0.15 | 0.38 | X69 judging (Emerald Rapids) |

The total is USD 0.38, an estimated gross figure. With the earlier experiments the campaign has
spent USD 15.01 of its USD 20 cap. Teardown was verified on 2026-11-24 at 16:55 UTC:
- every instance above was deleted after its ID was matched;
- the firewall rule, subnet and network were then deleted after their IDs were matched;
- the project lists only the `default` network;
- every watchdog was stopped.
