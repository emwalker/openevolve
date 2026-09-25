"""Finite, explicitly nominated parent draws shared across island copies."""

from copy import deepcopy
from threading import Lock


class ParentTrials:
    def __init__(self, key, values):
        if key is None and not values:
            self.enabled = False
            self._state = None
            return
        if (
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError("parent trials require a metadata key and distinct nonempty values")
        self.enabled = True
        self._state = {"key": key, "values": list(values), "draws": []}
        self._lock = Lock()

    def report(self):
        return deepcopy(self._state)

    def validate(self, programs, group_of):
        if self.enabled:
            self._pending(programs, group_of)

    def restore(self, state):
        if state is None:
            if self.enabled:
                self._state["draws"] = []
            else:
                self._state = None
            return
        if self.enabled and any(state[key] != self._state[key] for key in ("key", "values")):
            raise ValueError("parent trial configuration differs from the checkpoint")
        values = [draw["value"] for draw in state["draws"]]
        if len(set(values)) != len(values) or set(values) - set(state["values"]):
            raise ValueError("invalid parent trial draws in checkpoint")
        self._state = deepcopy(state)

    def draw(self, programs, local_ids, island, group_of, breedable):
        """Reserve one local trial at selection time, before worker submission."""
        if not self.enabled:
            return None
        with self._lock:
            candidates = self._pending(programs, group_of)
            for value, copies in candidates:
                local = sorted(p.id for p in copies if p.id in local_ids)
                if not local:
                    continue
                program = programs[local[0]]
                self._state["draws"].append(
                    {
                        "value": value,
                        "program_id": program.id,
                        "island": island,
                        "normally_eligible": program.id in breedable(list(local_ids)),
                        "metrics": dict(program.metrics),
                    }
                )
                return program
        return None

    def _pending(self, programs, group_of):
        drawn = {draw["value"] for draw in self._state["draws"]}
        pending = []
        for value in self._state["values"]:
            if value in drawn:
                continue
            copies = [p for p in programs.values() if p.metadata.get(self._state["key"]) == value]
            if not copies:
                raise ValueError(f"missing nominated parent: {value}")
            if len({(p.code, group_of(p)) for p in copies}) != 1:
                raise ValueError(f"ambiguous nominated parent: {value}")
            pending.append((value, copies))
        return pending
