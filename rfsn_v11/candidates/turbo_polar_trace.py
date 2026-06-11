from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class TurboPolarTraceEvent:
    event_type: str
    block_index: int
    timestamp_ns: int = 0
    bytes_transferred: int = 0


@dataclass
class TurboPolarTelemetry:
    cache_backend_used: str = "turbo_polar_k_only"
    real_cache_used: bool = True
    prefill_polar_encode_events: int = 0
    decode_polar_fetch_events: int = 0
    cache_bytes_written_actual: int = 0
    cache_bytes_read_actual: int = 0
    fallback_used: bool = False
    events: List[TurboPolarTraceEvent] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cache_backend_used": self.cache_backend_used,
            "real_cache_used": self.real_cache_used,
            "prefill_polar_encode_events": self.prefill_polar_encode_events,
            "decode_polar_fetch_events": self.decode_polar_fetch_events,
            "cache_bytes_written_actual": self.cache_bytes_written_actual,
            "cache_bytes_read_actual": self.cache_bytes_read_actual,
            "fallback_used": self.fallback_used,
            "event_count": len(self.events),
        }
