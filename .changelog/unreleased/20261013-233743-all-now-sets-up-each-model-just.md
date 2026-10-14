---
type: changed
breaking: false
audience: user
commits: 
---
`all` now sets up each model just before benchmarking it, and a setup failure is reported for that model instead of stopping the run. The command exits 1 when any model failed or was skipped and 130 when the run was aborted.
