---
type: fixed
breaking: false
audience: user
commits: 
---
Von no longer hangs on its first request. Its setup now downloads the model weights, and benchmark runs load them with Hugging Face networking turned off, so a slow or offline network can no longer stall a full run.
