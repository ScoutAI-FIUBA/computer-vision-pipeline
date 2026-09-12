from __future__ import annotations

from pathlib import Path


def _join_path_parts(parts: tuple[str, ...]) -> str:
    if not parts:
        return ""
    if parts[0] == "/":
        return str(Path("/").joinpath(*parts[1:]))
    return str(Path(*parts))


def _match_segment(pattern_part: str, value_part: str):
    if pattern_part == 'match{}' and value_part.startswith('match'):
        return [value_part.removeprefix('match')]
    if pattern_part == '{}.mp4' and value_part.endswith('.mp4'):
        return [value_part[:-4]]
    if '{}' in pattern_part:
        prefix, suffix = pattern_part.split('{}', 1)
        if value_part.startswith(prefix) and value_part.endswith(suffix):
            end = len(value_part) - len(suffix) if suffix else None
            return [value_part[len(prefix):end]]
    if pattern_part == value_part:
        return []
    return None


def _match_parts(pattern_parts: tuple[str, ...], value_parts: tuple[str, ...], pattern_index: int = 0, value_index: int = 0):
    if pattern_index == len(pattern_parts) and value_index == len(value_parts):
        return []
    if pattern_index == len(pattern_parts) or value_index == len(value_parts):
        return None

    pattern_part = pattern_parts[pattern_index]
    if pattern_part == '{}':
        remaining_pattern_parts = len(pattern_parts) - pattern_index - 1
        max_end = len(value_parts) - remaining_pattern_parts
        for end in range(value_index + 1, max_end + 1):
            remaining = _match_parts(pattern_parts, value_parts, pattern_index + 1, end)
            if remaining is not None:
                captured = _join_path_parts(value_parts[value_index:end])
                return [captured, *remaining]
        return None

    captures = _match_segment(pattern_part, value_parts[value_index])
    if captures is None:
        return None

    remaining = _match_parts(pattern_parts, value_parts, pattern_index + 1, value_index + 1)
    if remaining is None:
        return None
    return [*captures, *remaining]

try:
    import parse as parse
except ImportError:
    class _FallbackParse:
        @staticmethod
        def parse(pattern: str, value: str):
            pattern_path = Path(pattern)
            value_path = Path(value)
            pattern_parts = pattern_path.parts
            value_parts = value_path.parts

            captures = _match_parts(pattern_parts, value_parts)
            if captures is None:
                raise ValueError(f"pattern mismatch: {pattern!r} vs {value!r}")

            return tuple(captures)


    parse = _FallbackParse()
