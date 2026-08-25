# caustica thermal report — M18 evidence — 1 MHz bowl into a skin/brain phantom, 30 s on + 60 s off

CEM43 dose and ITRUSST threshold summary of one Pennes solve (`caustica-thermal/1`), generated 2026-08-24T09:32:30+00:00.

> **Research use only. caustica computes a modelled thermal dose from a modelled acoustic field; it is not a clinical decision tool and has no regulatory clearance. No number in this report is a medical claim, a treatment plan, or a safety clearance for an exposure of any living subject. Every value is the output of a numerical model of an idealised medium and is bounded by the tissue properties, the acoustic field and the discretisation it was given; the ITRUSST limits are quoted as published thresholds, and quoting them is not a statement that this simulation is an adequate basis for any decision about a person.**

## Verdict

|  |  |
|---|---|
| overall | **EXCEEDED** |
| dose (CEM43) | EXCEEDED |
| rise ≤ 2 °C | EXCEEDED |
| basis | per-tissue ITRUSST classes, each tissue against its own CEM43 limit |

## Temperature

|  |  |
|---|---|
| baseline [°C] | 37 |
| peak over the whole history [°C] | 54.93 |
| peak rise ΔT [°C] | 17.93 |
| peak at the END of the solve [°C] | 39.32 |
| mean at the END of the solve [°C] | 37.83 |

The peak is the per-voxel maximum over every step INCLUDING the internal sub-steps, not the endpoint: a sonication that has already cooled still did its damage on the way.

## Dose

|  |  |
|---|---|
| peak CEM43 [equivalent minutes] | 472.5 |
| mean CEM43 [equivalent minutes] | 0.2355 |
| voxels with any dose | 102400 / 102400 |

## ITRUSST thresholds

| threshold | limit | volume above | fraction | verdict |
|---|---|---|---|---|
| CEM43 (brain) | 2 equivalent minutes at 43 C | 55.13 mm³ (966 vox) | 0.943% | EXCEEDED |
| CEM43 (bone) | 16 equivalent minutes at 43 C | 14.15 mm³ (248 vox) | 0.242% | EXCEEDED |
| CEM43 (skin) | 21 equivalent minutes at 43 C | 10.67 mm³ (187 vox) | 0.183% | EXCEEDED |
| temperature rise (non-thermal line) | 2 C | 1159 mm³ (20303 vox) | 19.8% | EXCEEDED |

*ITRUSST consensus on biophysical safety for transcranial ultrasound stimulation, Brain Stimulation (2025): CEM43 <= 2 (brain), <= 16 (bone), <= 21 (skin); dT <= 2 C as the non-thermal criterion.* Each CEM43 row above reads the WHOLE volume against that limit — it is the as-if question ("if this were brain, would 2 CEM43 be crossed anywhere?"). The per-tissue table below is the answer for the tissue that is actually there. The rise row is the consensus' NON-THERMAL criterion: a deliberate ablation is expected to cross it.

## Per tissue

| id | tissue | ITRUSST class | size | peak T [°C] | peak ΔT [°C] | peak CEM43 | limit | over limit | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Skin | skin | 182.6 mm³ (3200 vox) | 47.7 | 10.7 | 4.024 | 21 | 0 | PASS |
| 2 | Brain | brain | 5661 mm³ (99200 vox) | 54.93 | 17.93 | 472.5 | 2 | 916 | EXCEEDED |

The ITRUSST class column is a match on the tissue NAME, printed so it can be checked; override it with `tissue_classes={id: 'brain'|'bone'|'skin'}`.

## Run

|  |  |
|---|---|
| scheme | `pennes-fd-explicit/1` |
| backend | numpy |
| boundary | insulated |
| heat source Q | HeatingSource(f0_only) -> none |
| perfusion | on |
| time | 89.92 s = 544 × 0.1653 s (× 1 sub-step(s)) |
| phases (source on / off) | 29.92 s with Q=HeatingSource(f0_only) → 60 s with Q=none |
| stability bound dt [s] | 0.1837 |
| grid | 40×40×64 @ 0.385 mm |

## Environment

|  |  |
|---|---|
| caustica | 0.1.0.dev0 @ 217d21ea434d |
| host | LOQ |
| python / numpy / scipy | 3.12.10 / 2.2.6 / 1.15.3 |
| platform | Windows-11-10.0.26200-SP0 |
| resolved backend | numpy |
| GPU | — (no CUDA device) |

## Caveats

- acoustics: westervelt on (56, 56, 80) at 0.385 mm (270 steps, converged period 28); peak |p| = 1.938 MPa in the recorded interior.
- heating: f0_only at harmonics (1,) (the second harmonic WAS recorded and deliberately not heated — the M18 honesty contract); Q_max = 1.858e+07 W/m^3, 1091 mW absorbed in the region.
- produced by tests/test_thermal_e2e.py::evidence_run — the same chain the e2e test runs, one size up.

## Files

- `thermal.json` — this report, machine-readable (`caustica-thermal/1`), including the liability note above verbatim

---

Research use only. caustica computes a modelled thermal dose from a modelled acoustic field; it is not a clinical decision tool and has no regulatory clearance. No number in this report is a medical claim, a treatment plan, or a safety clearance for an exposure of any living subject. Every value is the output of a numerical model of an idealised medium and is bounded by the tissue properties, the acoustic field and the discretisation it was given; the ITRUSST limits are quoted as published thresholds, and quoting them is not a statement that this simulation is an adequate basis for any decision about a person.
