# Conductor Setup (one-time per machine)

Read this only when `conductor whoami` fails. Once it succeeds, the machine is
configured for good and you never need this file again.

A healthy `conductor whoami` prints `email`, `short_id`, and available `teams`.
Anything else maps to one of the cases below.

## A. `command not found`

The CLI isn't installed. Install it (prefer the newest version; 3.2.0 is current):

```bash
pip install 'amd_conductor_cli==3.2.0' \
  --extra-index-url https://mkmartifactory.amd.com/artifactory/api/pypi/hw-orc3pypi-prod-local/simple/ \
  --trusted-host mkmartifactory.amd.com
```

## B. Keyring / D-Bus traceback

A traceback ending in `Failed to create the collection`, `SecretStorage`, or
`PromptDismissedException` means the CLI probes the OS keyring at import time and
crashes on a headless machine with no unlocked secret service. Force it onto its
environment-variable config path with keyring's built-in `fail` backend (no extra
install needed):

```bash
export PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring
```

Persist it by adding that line to `~/.bashrc` and `~/.zshrc`. (`keyrings.alt`
plaintext backends do not help — a fresh empty keyring makes the CLI crash reading
its own `verbosity_level`; the `fail` backend cleanly routes all config to env vars.)

## C. `401` error

Auth isn't configured (or was rejected). The CLI reads credentials from environment
variables. Ask the user to add these to `~/.bashrc` and `~/.zshrc`, then open a new
shell:

```bash
export ATS_URL=https://conductor.amd.com
export AMD_EMAIL=<first.last>@amd.com
export ATS_SECRET=<your-secret>
```

Never read, print, or echo the secret. It's the user's to set; you only need to know
that a `401` means it's missing or wrong and to remind them to configure it.

## D. Cannot resolve `conductor.amd.com`

DNS/VPN is down. Ask the user to connect the AMD VPN, then retry.

## Note on shell rc placement

Non-interactive shells (`bash -c`, `bash -lc`) skip `~/.bashrc` after its
"if not running interactively, return" guard, so exports placed below that guard
only load in interactive terminals. That's fine for normal use. If the vars must
also reach non-interactive scripts, put them above the guard or in `~/.profile`.
