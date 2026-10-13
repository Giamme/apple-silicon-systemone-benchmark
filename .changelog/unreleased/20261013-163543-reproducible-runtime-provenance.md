---
type: added
breaking: false
audience: user
commits:
---
New native model runs use pinned local weight snapshots, locked Python environments, and an explicit MLX free-buffer cache budget. JSON, CSV, Markdown, and offline HTML reports distinguish configured model and runtime pins from observed backend metadata, including load and inference failures.
