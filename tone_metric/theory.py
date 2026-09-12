from __future__ import annotations

"""Executable reference layer for the mathematical paper.

This module mirrors the formal derivation in
"From Metric States to Fractal Temporal Hierarchies: A Combinatorial Foundation
for Tone-Metric Analysis".  The operational analyzer does not need to recompute
Pascal's triangle in order to obtain the same recursive boundary sequence, but
these helpers make every derivational step explicit, testable, and traceable.

The functions here are intentionally pure and side-effect free.  They must never
modify score parsing, event timing, recursive Levels, waves, pivots, or trees.
"""

from fractions import Fraction
from math import comb
from typing import Iterable, Sequence


def validate_metric_word(word: str) -> str:
    normalized = "".join(str(word).split()).upper()
    if any(ch not in {"T", "A"} for ch in normalized):
        raise ValueError("Metric words may contain only T and A.")
    return normalized


def metric_word_path(word: str) -> list[tuple[int, int]]:
    """Definition 1 / Proposition 1: map a T/A word to its Pascal lattice path.

    A advances (i,j)->(i+1,j); T advances (i,j)->(i+1,j+1).
    The returned path includes the origin and every successive vertex.
    """
    word = validate_metric_word(word)
    i = j = 0
    out = [(0, 0)]
    for symbol in word:
        i += 1
        if symbol == "T":
            j += 1
        out.append((i, j))
    return out


def metric_word_endpoint(word: str) -> tuple[int, int]:
    word = validate_metric_word(word)
    return len(word), word.count("T")


def binomial_class_cardinality(n: int, k: int) -> int:
    """Equation (4): cardinality of the Pascal class containing n-length words with k T states."""
    if n < 0 or k < 0 or k > n:
        raise ValueError("Require integers with 0 <= k <= n.")
    return comb(n, k)


def repeated_meter_word(q: int, r: int) -> str:
    """Equations (2)-(3): [T A^(q-1)]^r."""
    if q < 1 or r < 0:
        raise ValueError("q must be >= 1 and r must be >= 0.")
    return ("T" + "A" * (q - 1)) * r


def _base_p_digits(value: int, p: int) -> list[int]:
    if value < 0:
        raise ValueError("Base-p digits require a non-negative integer.")
    if p < 2:
        raise ValueError("p must be >= 2.")
    if value == 0:
        return [0]
    out: list[int] = []
    while value:
        out.append(value % p)
        value //= p
    return out


def is_prime(p: int) -> bool:
    if p < 2:
        return False
    if p % 2 == 0:
        return p == 2
    d = 3
    while d * d <= p:
        if p % d == 0:
            return False
        d += 2
    return True


def lucas_binomial_mod(n: int, k: int, p: int) -> int:
    """Theorem 1: compute C(n,k) mod prime p using Lucas's digitwise product."""
    if not is_prime(p):
        raise ValueError("Lucas congruence requires prime p.")
    if n < 0 or k < 0 or k > n:
        return 0
    nd = _base_p_digits(n, p)
    kd = _base_p_digits(k, p)
    m = max(len(nd), len(kd))
    nd += [0] * (m - len(nd))
    kd += [0] * (m - len(kd))
    value = 1
    for ni, ki in zip(nd, kd):
        if ki > ni:
            return 0
        value = (value * comb(ni, ki)) % p
    return value


def pascal_binomial_mod(n: int, k: int, p: int) -> int:
    if p < 2:
        raise ValueError("p must be >= 2.")
    if n < 0 or k < 0 or k > n:
        return 0
    return comb(n, k) % p


def p_dilation_invariant(n: int, k: int, p: int, m: int = 1) -> bool:
    """Corollary 2: verify C(p^m n,p^m k) == C(n,k) (mod p)."""
    if m < 0:
        raise ValueError("m must be >= 0.")
    factor = p ** m
    return pascal_binomial_mod(factor * n, factor * k, p) == pascal_binomial_mod(n, k, p)


def scale_ray_point(p: int, m: int, n0: int = 1, k0: int = 0) -> tuple[int, int]:
    """Definition 2: v_m = p^m v_0."""
    if p < 2 or m < 0:
        raise ValueError("Require p >= 2 and m >= 0.")
    factor = p ** m
    return factor * n0, factor * k0


def temporal_projection(n: int | Fraction, tau: int | Fraction = 1, t0: int | Fraction = 0) -> Fraction:
    """Definition 3: Phi_(tau,t0)(n,k) = t0 + tau*n (k is combinatorial metadata)."""
    return Fraction(t0) + Fraction(tau) * Fraction(n)


def projected_scale_ray_time(
    p: int,
    m: int,
    *,
    n0: int = 1,
    tau: int | Fraction = 1,
    t0: int | Fraction = 1,
) -> Fraction:
    n, _ = scale_ray_point(p, m, n0=n0, k0=0)
    return temporal_projection(n, tau=tau, t0=t0)


def boundary_term(p: int, n: int) -> int:
    """Proposition 3: b_p(0)=1 and, for n>=1, b_p(n)=1+p^(n-1)."""
    if p < 2 or n < 0:
        raise ValueError("Require p >= 2 and n >= 0.")
    if n == 0:
        return 1
    return 1 + p ** (n - 1)


def boundary_sequence(p: int, count: int) -> list[int]:
    """Return the first ``count`` one-indexed boundary coordinates including b_p(0)."""
    if count < 0:
        raise ValueError("count must be >= 0.")
    return [boundary_term(p, n) for n in range(count)]


def boundary_sequence_recurrence(p: int, count: int) -> list[int]:
    """Equation (9), generated recursively rather than from the closed form."""
    if p < 2 or count < 0:
        raise ValueError("Require p >= 2 and count >= 0.")
    if count == 0:
        return []
    out = [1]
    if count == 1:
        return out
    out.append(2)
    while len(out) < count:
        out.append(p * out[-1] - (p - 1))
    return out


def validate_arity_schedule(schedule: Sequence[int]) -> tuple[int, ...]:
    """Definition 10: each explicitly modeled local arity p_j is prime."""
    out = tuple(int(x) for x in schedule)
    if not out:
        raise ValueError("Local arity schedule must contain at least one stage.")
    bad = [x for x in out if not is_prime(x)]
    if bad:
        raise ValueError(f"Every local arity must be prime; invalid value(s): {bad}")
    return out


def cumulative_scale_vector(schedule: Sequence[int]) -> tuple[int, ...]:
    """Definition 12: S_j=product_{i=1..j} p_i, preserving stage order."""
    schedule = validate_arity_schedule(schedule)
    product = 1
    out: list[int] = []
    for p in schedule:
        product *= p
        out.append(product)
    return tuple(out)


def theory_trace_for_word(word: str, p: int = 2) -> dict:
    """Compact auditable trace from metric word through Pascal class and p-adic status."""
    word = validate_metric_word(word)
    n, k = metric_word_endpoint(word)
    return {
        "metric_word": word,
        "path": metric_word_path(word),
        "endpoint": [n, k],
        "class_cardinality": binomial_class_cardinality(n, k),
        "modulus": p,
        "binomial_mod_p": pascal_binomial_mod(n, k, p),
        "lucas_mod_p": lucas_binomial_mod(n, k, p),
    }
