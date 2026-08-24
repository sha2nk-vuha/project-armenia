# Feature-aware inspection records and reporting

Each inspection records which Feature produced it (a `feature` column on
`inspections`), and stats and reports are scoped by Feature rather than pooling
all inspections together. The numeric score column becomes nullable to
accommodate Features (like Presence/Absence) that have no single anomaly score.

An "OK" from Anomaly Detection and an "OK" from Presence/Absence mean different
things, so pooling them into one pass-rate produces a misleading number. A
`feature` discriminator is cheap (it follows the existing PRAGMA-based lightweight
migration in `db.py`) and keeps each Feature's pass/fail statistics honest.
