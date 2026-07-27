# Security Policy

## Supported Versions

WatchGPU is currently in alpha. Security fixes are applied to the latest revision of the default branch.

## Reporting a Vulnerability

Please use GitHub's private vulnerability reporting when it is available for this repository. If the private form is unavailable, open a minimal issue asking the maintainer to establish a private channel; do not include vulnerability details in that issue. Do not publicly disclose vulnerabilities involving local privilege boundaries, socket authorization, unsafe filesystem handling, or process signalling.

Include a minimal description, affected revision, expected impact, and reproduction steps. Remove real hostnames, usernames, paths, PIDs, task names, GPU UUIDs, credentials, and proprietary training arguments.

## Scope

Relevant security properties include:

- control requests must remain within the same Unix UID;
- runtime paths must reject unsafe ownership and symlink substitution;
- stop/restart/reconciliation must never signal unrelated or training PIDs;
- leases must resist PID reuse and stale process identity;
- malformed local state or socket input must fail closed.

Resource fairness between processes running under the same UID is an operational policy issue, not a security isolation guarantee. See [docs/safety.md](docs/safety.md).
