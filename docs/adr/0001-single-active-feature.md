# One active Feature at a time

The app supports multiple inspection Features (Anomaly Detection, Presence/Absence),
but keeps exactly one model loaded at a time. Selecting a Feature swaps the active
model rather than holding both in memory.

We chose this over keeping both models resident because ONNX models plus their
runtimes are memory-heavy, an inspection station is configured for one job at a
time, and it preserves the existing single global session (`engine._current_session`).
The cost is a model (re)load on each Feature switch, which we accept as acceptable
for a station that changes jobs infrequently.
