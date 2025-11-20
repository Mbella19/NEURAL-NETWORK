"""Package initializer for src.

Importing this package configures the global logging system so that any module
under `src` emits structured traces immediately, as required by Phase 1.3's
observability goals.
"""

import os

from utils.logger import configure_logging, DISABLE_RESOURCE_MONITORING, log_resource_snapshot

# Initialize logging as soon as the project package is imported.
configure_logging()
if not DISABLE_RESOURCE_MONITORING:
    try:
        log_resource_snapshot("bootstrap")
    except Exception:
        pass

__all__ = ["configure_logging", "log_resource_snapshot", "DISABLE_RESOURCE_MONITORING"]
