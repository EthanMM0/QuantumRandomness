"""
Quantum Randomness mk1

An "omnipresent" quantum‑like randomness engine inspired by my quantum theory:
- Multiple independent sub‑engines evolve (optionally in threads) with diverse random ops
- Sub‑engines maintain *pending* state; **final value only resolves on observation** (press Enter)
- Optional environment entropy (time, PID, active thread count, OS randomness, psutil when available)
- Supports "gravity" and "environmental modes" (vacuum/cold/room) that bias evolution slightly
- Safe in thread‑limited sandboxes via SINGLE_THREAD_MODE fallback (no background threads)

!!! This is *not* cryptographically secure and is for experimentation/education !!!
"""
from __future__ import annotations

import os
import time
import math
import random
import secrets
import threading
from dataclasses import dataclass, field
from typing import List, Callable, Optional

# ------------------------ Config ------------------------
NUM_ENGINES: int = 4                 # logical sub‑engines ("universes")
SINGLE_THREAD_MODE: bool = True      # set True for restricted/thread‑limited envs
MUTATOR_SLEEP_MIN = 0.0005
MUTATOR_SLEEP_MAX = 0.004
DECIMAL_PRECISION = 4
MAX_MAGNITUDE = 1e12
DEFAULT_POSSIBILITIES = 7            # internal "many futures" per engine on observe

# optional psutil for richer entropy (safe import)
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

# --------------------- Symbol mapping -------------------
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
SYMBOLS  = "!@#$%^&*()_+-=[]\\{}|;':\",./<>?~`"

# Deterministic mapping (A=1..Z=26, then symbols continue) so values are readable
CHAR_MAP = {ch: i+1 for i, ch in enumerate(ALPHABET)}
start = len(CHAR_MAP)
for j, ch in enumerate(SYMBOLS, start=start):
    CHAR_MAP[ch] = j+1

# ----------------------- Utilities ----------------------

def safe_div(a: float, b: float) -> float:
    return a / (b if b != 0 else 1e-9)

def now_entropy() -> int:
    # nanosecond clock + pid + active threads + a few os random bytes
    base = time.perf_counter_ns() ^ os.getpid() ^ threading.active_count()
    rnd = int.from_bytes(os.urandom(8), 'little', signed=False)
    if psutil:
        try:
            base ^= int(psutil.cpu_percent(interval=0) * 1_000_000)
            base ^= int(psutil.virtual_memory().available % (1<<31))
            base ^= int(psutil.Process().num_threads())
        except Exception:
            pass
    return base ^ rnd

def complex_transform(x: float) -> float:
    a = abs(float(x)) + 1e-12
    p = (math.log1p(a) + 1.0) * (0.5 + random.random())
    term1 = a ** min(3.0, max(0.1, p))
    term2 = math.sin(a % (2*math.pi)) + math.cos((a+1) % (2*math.pi))
    term3 = math.log1p(a) if a > 1 else 0.0
    term4 = (1 + (a % 7)) * (1 + (now_entropy() & 0xFF))
    return float(term1 * (1 + 0.1*term2) + term3 + term4)

def enforce_constraints(v: float) -> float:
    if not math.isfinite(v) or v == 0:
        v = 1e-9
    if v <= -1:
        v = abs(v) + 1.0
    if abs(v) > MAX_MAGNITUDE:
        v = math.copysign(MAX_MAGNITUDE * (0.5 + random.random()*0.5), v)
    return float(v)

# ------------- Environment / Conceptual knobs -----------
@dataclass
class Context:
    gravity: float = 1.0            # >1 compresses spread (like stronger gravity)
    temperature: str = "room"        # "room" | "cold" | "vacuum" | "vacuum_cold"

    def bias(self) -> float:
        b = 1.0
        if self.temperature == "cold":
            b *= 0.95
        elif self.temperature == "vacuum":
            b *= 1.02
        elif self.temperature == "vacuum_cold":
            b *= 0.98
        return max(0.5, min(1.5, b * self.gravity))

# --------------------- Sub‑engine ------------------------
@dataclass
class SubEngine:
    primary: float
    secondary: float
    letter_seed: int
    pending_ops: List[Callable[[float, float], float]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    thread: Optional[threading.Thread] = None

    def schedule_random_op(self) -> None:
        dec = round(random.random(), DECIMAL_PRECISION+2)
        choice = random.choice(["add","sub","mul","div","chars","complex","concat"])

        def op(p: float, s: float) -> float:
            if choice == "add":
                return p + s*dec + random.random()
            if choice == "sub":
                return p - s*dec
            if choice == "mul":
                return p * (1 + dec*random.random())
            if choice == "div":
                return safe_div(p, (s % 7) + 1 + dec)
            if choice == "chars":
                # concatenate random letters/symbols as number and mix in
                length = random.randint(0, 5)
                if length:
                    # deterministic mapping => purely numeric text
                    allkeys = list(CHAR_MAP.keys())
                    chosen = ''.join(random.choice(allkeys) for _ in range(length))
                    val = int(''.join(str(CHAR_MAP[ch]) for ch in chosen))
                    return p + val * (random.random()+0.001)
                return p
            if choice == "complex":
                return p + complex_transform(s) * dec
            if choice == "concat":
                ssec = str(int(abs(s)))[-6:]
                slet = str(self.letter_seed)
                try:
                    combo = int(ssec + slet)
                except Exception:
                    combo = secrets.randbelow(1_000_000) + 1
                return p + combo * random.random()
            return p

        self.pending_ops.append(op)

    def tick(self, ctx: Context) -> None:
        # generate 1..3 new ops per tick
        for _ in range(1 + secrets.randbelow(3)):
            self.schedule_random_op()
        # evolve secondary a bit (conceptual coupling)
        self.secondary = enforce_constraints(self.secondary + random.random()*ctx.bias())

    def resolve(self, ctx: Context) -> float:
        # Apply all pending ops at once when observed (measurement collapse analogue)
        with self.lock:
            p, s = self.primary, self.secondary
            while self.pending_ops:
                op = self.pending_ops.pop(0)
                p = enforce_constraints(op(p, s))
                # sometimes refresh secondary from transformed primary
                if random.random() < 0.3:
                    s = enforce_constraints(complex_transform(p) * (0.5 + random.random()))
            # gravity/temperature bias
            p *= ctx.bias()
            # store updated state
            self.primary, self.secondary = p, s
            return p

# --------------------- Omni Engine -----------------------
class OmniQuantumEngine:
    def __init__(self, n: int = NUM_ENGINES, ctx: Optional[Context] = None):
        self.ctx = ctx or Context()
        self.engines: List[SubEngine] = []
        rngboost = now_entropy() ^ int.from_bytes(secrets.token_bytes(8), 'little')
        
        # FIX: ensure seeding uses integer XOR, not float
        seed_val = int(random.random() * (1 << 32)) ^ rngboost
        random.seed(seed_val)
        
        for _ in range(max(1, n)):
            se = SubEngine(
                primary=float(secrets.randbelow(10**6)) + random.random(),
                secondary=float(secrets.randbelow(10**6)) + random.random(),
                letter_seed=int.from_bytes(secrets.token_bytes(2), 'little')
            )
            self.engines.append(se)
        self._running = False


    def start(self) -> None:
        if SINGLE_THREAD_MODE:
            self._running = True
            return
        if self._running:
            return
        self._running = True
        for se in self.engines:
            t = threading.Thread(target=self._mutator_loop, args=(se,), daemon=True)
            se.thread = t
            t.start()

    def _mutator_loop(self, se: SubEngine) -> None:
        while self._running:
            time.sleep(random.uniform(MUTATOR_SLEEP_MIN, MUTATOR_SLEEP_MAX))
            se.tick(self.ctx)

    def stop(self) -> None:
        self._running = False
        for se in self.engines:
            if se.thread:
                se.thread.join(timeout=1.0)

    def _advance_if_single_thread(self) -> None:
        if SINGLE_THREAD_MODE:
            # Advance based on elapsed time to mimic background evolution
            steps = 1 + (now_entropy() & 0x3)
            for _ in range(steps):
                for se in self.engines:
                    se.tick(self.ctx)

    def observe(self, possibilities: int = DEFAULT_POSSIBILITIES) -> str:
        """Generate multiple hidden possibilities per engine and reveal one shadow value.
        Returns a numeric string (letters/symbols are mapped to digits internally).
        """
        self._advance_if_single_thread()
        shadows: List[float] = []
        for se in self.engines:
            for _ in range(max(1, possibilities)):
                val = se.resolve(self.ctx)
                # add tiny jitter from fresh entropy to represent micro‑contexts
                val = enforce_constraints(val + (now_entropy() & 0xFFFF) * 1e-6 * random.random())
                shadows.append(val)
        chosen = random.choice(shadows) if shadows else 0.0
        # prepend a short numeric token derived from a few random chars (shown as digits)
        prefix_len = secrets.randbelow(5)  # 0..4
        if prefix_len:
            keys = list(CHAR_MAP.keys())
            token = ''.join(str(CHAR_MAP[random.choice(keys)]) for _ in range(prefix_len))
            return f"{token}{int(round(chosen))}"
        return str(int(round(chosen)))

# ------------------------- CLI ---------------------------

def main() -> None:
    print("OmniQuantumEngine — press Enter to observe (type 'q' + Enter to quit).")
    print(f"mode: {'single‑thread' if SINGLE_THREAD_MODE else 'multi‑thread'} | engines: {len(range(NUM_ENGINES))}")
    engine = OmniQuantumEngine(n=NUM_ENGINES, ctx=Context(gravity=1.0, temperature='room'))
    engine.start()
    try:
        while True:
            user = input()
            if user.strip().lower() == 'q':
                print('Stopping engine...')
                break
            print('Observed token:', engine.observe())
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt — shutting down...')
    finally:
        engine.stop()

# ------------------------ Tests -------------------------

def _tests() -> None:
    # Basic sanity tests (do not change unless wrong)
    e = OmniQuantumEngine(n=2)
    e.start()
    a = e.observe()
    b = e.observe()
    assert isinstance(a, str) and a.strip('-').isdigit(), "observe() must return numeric string"
    assert isinstance(b, str) and b.strip('-').isdigit(), "observe() must return numeric string"
    # Different calls should often differ (not guaranteed), so we allow equality but check many samples
    samples = {e.observe() for _ in range(10)}
    assert len(samples) >= 2, "observations should vary across calls"
    # Single‑thread fallback should work without threads
    global SINGLE_THREAD_MODE
    SINGLE_THREAD_MODE = True
    e2 = OmniQuantumEngine(n=1)
    e2.start()
    _ = e2.observe()
    e2.stop()
    print("All tests passed.")

if __name__ == '__main__':
    # Run as script by default
    main()
    # Uncomment to run tests manually:
    # _tests()
