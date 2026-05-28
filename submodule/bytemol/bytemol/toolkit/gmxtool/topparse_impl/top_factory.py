# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
from uuid import uuid4

logger = logging.getLogger(__name__)


class Factory:
    """Keyed singleton (multiton) base class used by the GROMACS topology parser.

    This is called a "Factory" historically, but its behavior is closer to a
    *per-scope singleton registry*:

    - Each subclass of :class:`Factory` has **one shared instance per `uuid`**.
    - The `uuid` acts as a *scope identifier* (e.g., a parsed topology system).

    Why this exists in `gmxtool/topparse`:

    - Classes like `TopoDefaults` and `TopoAtomTypes` need to be shared across
      many objects (molecules/records) belonging to the same parsed system.
    - Instead of threading a context object through every constructor, code can
      call `TopoAtomTypes(system_uuid)` anywhere and get the same instance.
    - Parsing multiple systems in one Python process remains isolated because
      each system uses a different `uuid`.

    Lifecycle notes:

    - Instances are cached in :attr:`_instances`.
    - :meth:`remove` allows explicit cleanup for a given `uuid`.
    """

    _instances = {}

    def __new__(cls, uuid=None):
        """Return the cached instance for (cls, uuid), creating it if needed."""
        k = uuid4() if uuid is None else uuid
        if (cls, k) not in cls._instances:
            instance = super().__new__(cls)
            instance.uuid = k
            instance._initialized = False
            cls._instances[(cls, k)] = instance
        return cls._instances[(cls, k)]

    @classmethod
    def remove(cls, uuid):
        """Remove the cached instance for this subclass and `uuid` (if present)."""
        if (cls, uuid) in cls._instances:
            del cls._instances[(cls, uuid)]
