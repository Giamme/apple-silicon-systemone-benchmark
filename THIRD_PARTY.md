# Third-party projects

This repository provides a benchmark harness. It downloads no model code or weights into Git and does not relicense them. Built-in setup currently fetches:

| Candidate | Code or package | Weights |
| --- | --- | --- |
| Laya | [laya on PyPI](https://pypi.org/project/laya/) | [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) |
| Von | [wfzyx/von](https://github.com/wfzyx/von) | [wfzyx/von](https://huggingface.co/wfzyx/von) |
| Kev | [jaredpalmer/kev](https://github.com/jaredpalmer/kev) | [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) and its linked base model |
| Rizzo Flow | [Rizzo-AI-Academy/rizzo-flow](https://github.com/Rizzo-AI-Academy/rizzo-flow) | Downloaded from the upstream model repository by the Rizzo Flow CLI |

Each dependency and weight file retains its own terms. Check its current upstream license before distributing a bundle, container, or modified copy. The Apache-2.0 license in this repository applies to this harness and its synthetic examples.
