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
