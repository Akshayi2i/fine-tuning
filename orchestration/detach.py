"""Every long job on the pod runs detached, in tmux — whoever starts it, however.

A job started in an SSH terminal dies with that terminal: shutting the laptop,
letting it sleep or losing Wi-Fi ends the SSH session, the session's processes
get SIGHUP, and a training run hours in is gone. ``scripts/pod_run.sh`` avoids
that, but only if it is remembered every time. This makes it the only way:
on a RunPod pod, a long command that is not already inside tmux re-launches
itself through ``pod_run.sh`` and returns at once, telling you how to watch it.

Off the pod (a laptop, CI) nothing changes: commands run in the foreground as
they always did. Inside tmux, or inside a ``pod_run.sh`` run, they also run in
the foreground — they are already safe, and re-detaching would start a second
copy. ``FIDEON_NO_DETACH=1`` runs a command in the foreground on the pod anyway,
for the rare case that is wanted; it is never set by this code.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)

POD_RUN = Path(__file__).resolve().parent.parent / "scripts" / "pod_run.sh"

#: Set by pod_run.sh inside every run it starts.
DETACHED_ENV = "FIDEON_DETACHED"
#: Opt out, for one command, on the pod.
NO_DETACH_ENV = "FIDEON_NO_DETACH"
#: RunPod sets this in every pod.
POD_ENV = "RUNPOD_POD_ID"


def on_pod() -> bool:
    return bool(os.environ.get(POD_ENV))


def already_safe() -> bool:
    """Inside a pod_run.sh run, or inside tmux the operator started themselves."""
    return os.environ.get(DETACHED_ENV) == "1" or bool(os.environ.get("TMUX"))


def should_detach() -> bool:
    return on_pod() and not already_safe() and os.environ.get(NO_DETACH_ENV) != "1"


def run_name(hint: str) -> str:
    """``finetune-20261003-141502`` — unique per start, valid for pod_run.sh."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{re.sub(r'[^A-Za-z0-9._-]+', '-', hint).strip('-') or 'job'}-{stamp}"


def detach(command: Sequence[str], hint: str) -> str:
    """Start ``command`` in a tmux session through pod_run.sh. Returns the run name.

    ``command`` is the full command line to run (``python -m ...``, ``bash ...``).
    """
    name = run_name(hint)
    subprocess.run(["bash", str(POD_RUN), "start", name, "--", *command], check=True)
    return name


def detach_if_needed(command: Sequence[str], hint: str) -> bool:
    """Re-launch ``command`` detached when on the pod and not already safe.

    Returns True when it did: the caller should exit at once, because the job now
    runs in tmux and running it here too would start it twice.
    """
    if not should_detach():
        return False
    name = detach(command, hint)
    print(
        f"\nOn the pod, long jobs run in tmux so a closed laptop or dropped Wi-Fi cannot stop "
        f"them. This one is running as '{name}'.\n"
        f"  status: bash scripts/pod_run.sh status {name}\n"
        f"  watch:  bash scripts/pod_run.sh attach {name}   (Ctrl-b then d to leave it running)\n"
        f"  log:    bash scripts/pod_run.sh tail {name}\n"
    )
    return True


def detach_module_if_needed(module: str, argv: Sequence[str] | None, hint: str | None = None) -> bool:
    """:func:`detach_if_needed` for ``python -m <module> <argv>``."""
    import sys

    args = list(argv) if argv is not None else sys.argv[1:]
    return detach_if_needed([sys.executable, "-m", module, *args], hint or module.rsplit(".", 1)[-1])


def detach_script_if_needed(script: str | Path, argv: Sequence[str] | None, hint: str) -> bool:
    """:func:`detach_if_needed` for ``python <script> <argv>``."""
    import sys

    args = list(argv) if argv is not None else sys.argv[1:]
    return detach_if_needed([sys.executable, str(script), *args], hint)
