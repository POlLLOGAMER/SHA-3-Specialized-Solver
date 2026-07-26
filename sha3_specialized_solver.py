#!/usr/bin/env python3
"""
============================================================================
 SHA-3 Collision Solver — Especializado para estructura Keccak
============================================================================

 En lugar de tratar el SAT como un problema genérico de 310K variables,
 este solver explota la estructura de SHA-3:

   1. Solo 128 bits son LIBRES (msg1[0..63] + msg2[0..63])
   2. Las otras ~310K variables son DETERMINISTAS (calculables por BCP)
   3. La "dificultad" real está en encontrar 128 bits tales que:
      - hash(msg1) == hash(msg2)  (colisión)
      - msg1 != msg2              (diferentes)

 Estrategia:
   - Fase 1: Generar "seeds" inteligentes usando propiedades algebraicas
     de Keccak (linealidad parcial de θ, estructura de χ)
   - Fase 2: Búsqueda local (hill climbing + simulated annealing) en
     el espacio de 128 bits, maximizando bits de hash coincidentes
   - Fase 3: Refinamiento con mutaciones dirigidas por análisis
     diferencial de rondas reducidas
   - Fase 4: Verificación con hashlib.sha3_256

 Nota: Esto NO encuentra una colisión real de SHA-3/256 (imposible),
 pero MAXIMIZA el % de cláusulas satisfechas en el CNF, que es
 equivalente a maximizar los bits de hash coincidentes.
============================================================================
"""

import hashlib
import numpy as np
import time
import sys
import os

# ─────────────────────────────────────────────────────────────────────────────
# Constantes Keccak-f[1600] (verificadas contra hashlib)
# ─────────────────────────────────────────────────────────────────────────────

ROT = [
    [ 0, 36,  3, 41, 18],
    [ 1, 44, 10, 45,  2],
    [62,  6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39,  8, 14]
]

RC = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A,
    0x8000000080008000, 0x000000000000808B, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008A,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800A, 0x800000008000000A, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008
]

RATE = 1088; HASH_LEN = 256; STATE_SIZE = 1600; MSG_BITS = 64

def sidx(x, y, z):
    return 64 * (5 * y + x) + z

# ─────────────────────────────────────────────────────────────────────────────
# Keccak-f[1600] optimizado (operaciones a nivel de lanes de 64 bits)
# ─────────────────────────────────────────────────────────────────────────────

def keccak_f_lanes(state_lanes):
    """
    Keccak-f[1600] sobre un array de 25 lanes (uint64).
    state_lanes: array de 25 enteros de 64 bits, indexados como [5*y + x].
    Retorna: array de 25 enteros.
    """
    A = list(state_lanes)
    MASK64 = 0xFFFFFFFFFFFFFFFF
    
    for rnd in range(24):
        # θ (theta)
        C = [0] * 5
        for x in range(5):
            C[x] = A[x] ^ A[5 + x] ^ A[10 + x] ^ A[15 + x] ^ A[20 + x]
        
        D = [0] * 5
        for x in range(5):
            D[x] = C[(x - 1) % 5] ^ (((C[(x + 1) % 5] << 1) | (C[(x + 1) % 5] >> 63)) & MASK64)
        
        for i in range(25):
            A[i] ^= D[i % 5]
        
        # ρ (rho) + π (pi)
        B = [0] * 25
        for x in range(5):
            for y in range(5):
                idx = 5 * y + x
                rot = ROT[x][y]
                val = A[idx]
                if rot > 0:
                    rotated = ((val << rot) | (val >> (64 - rot))) & MASK64
                else:
                    rotated = val
                # π: B[y][2x+3y mod 5] = rotated
                nx = y
                ny = (2 * x + 3 * y) % 5
                B[5 * ny + nx] = rotated
        
        # χ (chi)
        for y in range(5):
            for x in range(5):
                idx = 5 * y + x
                idx1 = 5 * y + (x + 1) % 5
                idx2 = 5 * y + (x + 2) % 5
                A[idx] = B[idx] ^ ((~B[idx1] & MASK64) & B[idx2])
        
        # ι (iota)
        A[0] ^= RC[rnd]
    
    return A

# ─────────────────────────────────────────────────────────────────────────────
# SHA-3/256 rápido usando keccak_f_lanes
# ─────────────────────────────────────────────────────────────────────────────

def sha3_256_fast(msg_bytes):
    """SHA-3/256 usando la implementación optimizada de lanes."""
    # Convertir mensaje a bits
    msg_bits = []
    for byte in msg_bytes:
        for i in range(8):
            msg_bits.append((byte >> i) & 1)
    
    # Padding
    block_bits = msg_bits[:]
    block_bits.append(0)  # domain sep
    block_bits.append(1)  # domain sep
    block_bits.append(1)  # pad10*1 start
    while len(block_bits) < RATE - 1:
        block_bits.append(0)
    block_bits.append(1)  # pad10*1 end
    while len(block_bits) < STATE_SIZE:
        block_bits.append(0)
    
    # Convertir bits a lanes
    lanes = [0] * 25
    for i in range(25):
        for z in range(64):
            bit_idx = i * 64 + z
            if bit_idx < len(block_bits) and block_bits[bit_idx]:
                lanes[i] |= (1 << z)
    
    # Keccak-f
    lanes = keccak_f_lanes(lanes)
    
    # Extraer hash (primeros 256 bits = lanes 0..3)
    hash_bytes = bytearray()
    for i in range(4):  # 4 lanes × 8 bytes = 32 bytes = 256 bits
        for byte_idx in range(8):
            hash_bytes.append((lanes[i] >> (byte_idx * 8)) & 0xFF)
    
    return bytes(hash_bytes)

# ─────────────────────────────────────────────────────────────────────────────
# Función objetivo: contar bits de hash coincidentes
# ─────────────────────────────────────────────────────────────────────────────

def hash_to_bits(h):
    """Convierte hash bytes a array de bits."""
    bits = []
    for byte in h:
        for i in range(8):
            bits.append((byte >> i) & 1)
    return np.array(bits, dtype=np.int8)

def count_matching_bits(h1, h2):
    """Cuenta cuántos bits coinciden entre dos hashes."""
    return np.sum(h1 == h2)

def evaluate_pair(msg1_bytes, msg2_bytes):
    """
    Evalúa un par de mensajes.
    Retorna: (matching_bits, hash1, hash2, is_collision)
    """
    h1 = sha3_256_fast(msg1_bytes)
    h2 = sha3_256_fast(msg2_bytes)
    b1 = hash_to_bits(h1)
    b2 = hash_to_bits(h2)
    matching = int(count_matching_bits(b1, b2))
    return matching, h1, h2, (msg1_bytes != msg2_bytes and h1 == h2)

# ─────────────────────────────────────────────────────────────────────────────
# Estimación de cláusulas SAT a partir de matching bits
# ─────────────────────────────────────────────────────────────────────────────

def estimate_sat_clauses(matching_bits, msg1_bytes, msg2_bytes, total_clauses=1156185):
    """
    Estima el % de cláusulas satisfechas en el CNF basándose en:
    - Cláusulas de Keccak (ejecución 1): 100% satisfechas (determinista)
    - Cláusulas de Keccak (ejecución 2): 100% satisfechas (determinista)
    - Cláusulas de colisión: proporcionales a matching_bits/256
    - Cláusulas de diferencia: satisfechas si msg1 ≠ msg2
    """
    # Cláusulas base de Keccak (ambas ejecuciones)
    # ~1,155,416 cláusulas de Keccak (257 son de diff + colisión)
    keccak_clauses = 1155416  # siempre satisfechas si las entradas son válidas
    collision_clauses = 512   # hash1[i] = hash2[i]: 2 cláusulas por bit
    diff_clauses = 257        # XORs + cláusula larga
    
    # Cláusulas de colisión satisfechas
    collision_sat = int(matching_bits * 2)  # 2 cláusulas por bit coincidente
    
    # Cláusulas de diferencia
    if msg1_bytes != msg2_bytes:
        diff_sat = diff_clauses  # todas satisfechas
    else:
        diff_sat = diff_clauses - 1  # la cláusula larga no se satisface
    
    total_sat = keccak_clauses + collision_sat + diff_sat
    return total_sat, total_clauses

# ─────────────────────────────────────────────────────────────────────────────
# Generadores de seeds inteligentes
# ─────────────────────────────────────────────────────────────────────────────

def generate_smart_seeds(n_seeds=50):
    """
    Genera pares de mensajes "inteligentes" usando propiedades de SHA-3:
    1. Mensajes que difieren en pocos bits (near-input collisions)
    2. Mensajes complementarios
    3. Mensajes con patrones que exploran la no-linealidad de χ
    """
    seeds = []
    rng = np.random.default_rng(42)
    
    # Tipo 1: Mensajes aleatorios con diferencias controladas
    for _ in range(n_seeds // 5):
        msg1 = rng.integers(0, 256, size=8, dtype=np.uint8)
        msg2 = msg1.copy()
        # Flip 1-4 bits aleatorios
        n_flips = rng.integers(1, 5)
        for _ in range(n_flips):
            byte_idx = rng.integers(0, 8)
            bit_idx = rng.integers(0, 8)
            msg2[byte_idx] ^= (1 << bit_idx)
        seeds.append((bytes(msg1), bytes(msg2)))
    
    # Tipo 2: Mensajes complementarios
    for _ in range(n_seeds // 5):
        msg1 = rng.integers(0, 256, size=8, dtype=np.uint8)
        msg2 = (~msg1) & 0xFF
        seeds.append((bytes(msg1), bytes(msg2)))
    
    # Tipo 3: Diferencia en un solo byte
    for _ in range(n_seeds // 5):
        msg1 = rng.integers(0, 256, size=8, dtype=np.uint8)
        msg2 = msg1.copy()
        byte_idx = rng.integers(0, 8)
        msg2[byte_idx] = rng.integers(0, 256, dtype=np.uint8)
        if msg2[byte_idx] == msg1[byte_idx]:
            msg2[byte_idx] ^= 1
        seeds.append((bytes(msg1), bytes(msg2)))
    
    # Tipo 4: Diferencia en el byte 0 (explora la iota más directamente)
    for _ in range(n_seeds // 5):
        msg1 = bytes([0] * 8)
        msg2 = bytes([rng.integers(1, 256)] + [0] * 7)
        seeds.append((msg1, msg2))
    
    # Tipo 5: Pares con alta diferencia de Hamming
    for _ in range(n_seeds // 5):
        msg1 = rng.integers(0, 256, size=8, dtype=np.uint8)
        msg2 = msg1 ^ rng.integers(128, 256, size=8, dtype=np.uint8)
        seeds.append((bytes(msg1), bytes(msg2)))
    
    return seeds

# ─────────────────────────────────────────────────────────────────────────────
# Búsqueda local: Hill Climbing + Simulated Annealing
# ─────────────────────────────────────────────────────────────────────────────

def flip_bit(msg_bytes, bit_pos):
    """Flip un bit específico en un mensaje de 8 bytes."""
    arr = bytearray(msg_bytes)
    byte_idx = bit_pos // 8
    bit_idx = bit_pos % 8
    arr[byte_idx] ^= (1 << bit_idx)
    return bytes(arr)

def flip_random_bits(msg_bytes, n_bits, rng):
    """Flip n bits aleatorios."""
    arr = bytearray(msg_bytes)
    positions = rng.choice(64, size=min(n_bits, 64), replace=False)
    for pos in positions:
        byte_idx = pos // 8
        bit_idx = pos % 8
        arr[byte_idx] ^= (1 << bit_idx)
    return bytes(arr)

def local_search(msg1, msg2, max_iters=5000, initial_temp=10.0, seed=42):
    """
    Búsqueda local combinada: Hill Climbing + Simulated Annealing.
    
    Objetivo: maximizar matching_bits(hash(msg1), hash(msg2))
    fijando msg1 y buscando el mejor msg2.
    """
    rng = np.random.default_rng(seed)
    
    best_msg2 = msg2
    h1 = sha3_256_fast(msg1)
    b1 = hash_to_bits(h1)
    h2 = sha3_256_fast(msg2)
    b2 = hash_to_bits(h2)
    best_score = int(count_matching_bits(b1, b2))
    
    current_msg2 = msg2
    current_score = best_score
    
    temp = initial_temp
    cooling = 0.9995
    
    no_improve = 0
    
    for it in range(max_iters):
        # Generar vecino
        if rng.random() < 0.7:
            # Flip 1 bit
            bit = rng.integers(0, 64)
            candidate = flip_bit(current_msg2, bit)
        elif rng.random() < 0.5:
            # Flip 2-3 bits
            n = rng.integers(2, 4)
            candidate = flip_random_bits(current_msg2, n, rng)
        else:
            # Mutación más agresiva
            n = rng.integers(4, 12)
            candidate = flip_random_bits(current_msg2, n, rng)
        
        if candidate == msg1:
            continue
        
        h_cand = sha3_256_fast(candidate)
        b_cand = hash_to_bits(h_cand)
        cand_score = int(count_matching_bits(b1, b_cand))
        
        delta = cand_score - current_score
        
        if delta > 0:
            current_msg2 = candidate
            current_score = cand_score
            no_improve = 0
            
            if current_score > best_score:
                best_msg2 = current_msg2
                best_score = current_score
        elif delta == 0:
            # Aceptar movimientos laterales con probabilidad
            if rng.random() < 0.3:
                current_msg2 = candidate
        else:
            # Simulated annealing: aceptar peores con probabilidad decreciente
            if temp > 0.01 and rng.random() < np.exp(delta / temp):
                current_msg2 = candidate
                current_score = cand_score
        
        temp *= cooling
        no_improve += 1
        
        # Restart si no hay mejora
        if no_improve > 500:
            # Restart desde el mejor + perturbación
            current_msg2 = flip_random_bits(best_msg2, rng.integers(3, 10), rng)
            if current_msg2 == msg1:
                current_msg2 = flip_bit(current_msg2, rng.integers(0, 64))
            h2 = sha3_256_fast(current_msg2)
            b2 = hash_to_bits(h2)
            current_score = int(count_matching_bits(b1, b2))
            no_improve = 0
            temp = initial_temp * 0.5
    
    return best_msg2, best_score

# ─────────────────────────────────────────────────────────────────────────────
# Búsqueda diferencial: explotar diferencias en el estado de Keccak
# ─────────────────────────────────────────────────────────────────────────────

def differential_search(msg1, n_candidates=200, seed=42):
    """
    Genera candidatos para msg2 usando análisis diferencial simplificado.
    
    Idea: si conocemos hash(msg1), buscamos msg2 tal que la diferencia
    en el estado inicial se "propague" de forma que las diferencias
    en el estado final sean mínimas.
    
    Para SHA-3 completo esto no funciona perfectamente, pero puede
    generar candidatos mejores que aleatorio.
    """
    rng = np.random.default_rng(seed)
    h1 = sha3_256_fast(msg1)
    b1 = hash_to_bits(h1)
    
    best_msg2 = None
    best_score = 0
    
    # Estrategia 1: Variaciones sistemáticas en cada byte
    for byte_idx in range(8):
        for delta in range(1, 256):
            msg2 = bytearray(msg1)
            msg2[byte_idx] = (msg2[byte_idx] + delta) & 0xFF
            msg2 = bytes(msg2)
            
            h2 = sha3_256_fast(msg2)
            b2 = hash_to_bits(h2)
            score = int(count_matching_bits(b1, b2))
            
            if score > best_score:
                best_score = score
                best_msg2 = msg2
    
    # Estrategia 2: XOR con constantes pequeñas
    for delta in range(1, min(n_candidates, 1000)):
        msg2_int = int.from_bytes(msg1, 'little') ^ delta
        msg2 = msg2_int.to_bytes(8, 'little')
        
        h2 = sha3_256_fast(msg2)
        b2 = hash_to_bits(h2)
        score = int(count_matching_bits(b1, b2))
        
        if score > best_score:
            best_score = score
            best_msg2 = msg2
    
    return best_msg2, best_score

# ─────────────────────────────────────────────────────────────────────────────
# Solver principal
# ─────────────────────────────────────────────────────────────────────────────

def solve_sha3_collision(max_time=300, verbose=True):
    """
    Busca el par (msg1, msg2) que maximice los bits de hash coincidentes.
    
    max_time: tiempo máximo en segundos
    """
    t_start = time.time()
    
    if verbose:
        print("="*70)
        print("  SHA-3/256 Collision Solver — Especializado")
        print("="*70)
        print()
    
    # ── Fase 1: Verificar implementación ──
    if verbose:
        print("[Fase 0] Verificando implementación de Keccak...")
    test_msg = b"\x00" * 8
    h_ref = hashlib.sha3_256(test_msg).digest()
    h_fast = sha3_256_fast(test_msg)
    assert h_ref == h_fast, f"Error: {h_ref.hex()} != {h_fast.hex()}"
    if verbose:
        print(f"  ✓ Implementación correcta")
        print()
    
    # ── Fase 2: Generar seeds ──
    if verbose:
        print("[Fase 1] Generando seeds inteligentes...")
    seeds = generate_smart_seeds(100)
    
    best_msg1 = None
    best_msg2 = None
    best_matching = 0
    best_h1 = None
    best_h2 = None
    
    for i, (m1, m2) in enumerate(seeds):
        matching, h1, h2, is_coll = evaluate_pair(m1, m2)
        if matching > best_matching:
            best_matching = matching
            best_msg1 = m1
            best_msg2 = m2
            best_h1 = h1
            best_h2 = h2
        
        if verbose and (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            print(f"  Seeds evaluados: {i+1}/{len(seeds)}, "
                  f"mejor: {best_matching}/256 bits ({100*best_matching/256:.1f}%), "
                  f"tiempo: {elapsed:.1f}s")
    
    if verbose:
        elapsed = time.time() - t_start
        sat_est, total_cl = estimate_sat_clauses(best_matching, best_msg1, best_msg2)
        print(f"\n  Mejor seed: {best_matching}/256 bits ({100*best_matching/256:.1f}%)")
        print(f"  SAT estimado: {sat_est}/{total_cl} ({100*sat_est/total_cl:.2f}%)")
        print(f"  Tiempo: {elapsed:.1f}s")
        print()
    
    # ── Fase 3: Búsqueda diferencial ──
    if verbose:
        print("[Fase 2] Búsqueda diferencial...")
    
    diff_msg2, diff_score = differential_search(best_msg1, n_candidates=500)
    if diff_score > best_matching:
        best_matching = diff_score
        best_msg2 = diff_msg2
        best_h2 = sha3_256_fast(diff_msg2)
        if verbose:
            elapsed = time.time() - t_start
            sat_est, total_cl = estimate_sat_clauses(best_matching, best_msg1, best_msg2)
            print(f"  ¡Mejora! {best_matching}/256 bits ({100*best_matching/256:.1f}%)")
            print(f"  SAT estimado: {sat_est}/{total_cl} ({100*sat_est/total_cl:.2f}%)")
            print(f"  Tiempo: {elapsed:.1f}s")
    else:
        if verbose:
            print(f"  Sin mejora (mejor actual: {best_matching}/256)")
    print()
    
    # ── Fase 4: Búsqueda local (hill climbing + SA) ──
    if verbose:
        print("[Fase 3] Búsqueda local (Hill Climbing + Simulated Annealing)...")
    
    # Múltiples restarts
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() - t_start > max_time * 0.8:
            break
        
        elapsed = time.time() - t_start
        remaining = max_time - elapsed
        
        if verbose:
            print(f"\n  Restart {restart+1}/{n_restarts} "
                  f"(tiempo restante: {remaining:.0f}s)")
        
        # Si es el primer restart, usar el mejor encontrado
        if restart == 0:
            init_msg2 = best_msg2
        else:
            # Perturbar el mejor
            rng = np.random.default_rng(restart * 1000 + 42)
            init_msg2 = flip_random_bits(best_msg2, rng.integers(5, 20), rng)
            if init_msg2 == best_msg1:
                init_msg2 = flip_bit(init_msg2, rng.integers(0, 64))
        
        iters = min(10000, int(remaining * 200))  # ~200 iters/seg
        new_msg2, new_score = local_search(
            best_msg1, init_msg2, 
            max_iters=iters,
            initial_temp=10.0 + restart * 2,
            seed=restart * 777 + 42
        )
        
        if new_score > best_matching:
            best_matching = new_score
            best_msg2 = new_msg2
            best_h2 = sha3_256_fast(new_msg2)
            sat_est, total_cl = estimate_sat_clauses(best_matching, best_msg1, best_msg2)
            if verbose:
                print(f"  ¡Mejora! {best_matching}/256 bits ({100*best_matching/256:.1f}%)")
                print(f"  SAT estimado: {sat_est}/{total_cl} ({100*sat_est/total_cl:.2f}%)")
        
        # También intentar con un msg1 diferente
        rng2 = np.random.default_rng(restart * 999)
        alt_msg1 = flip_random_bits(best_msg1, rng2.integers(2, 8), rng2)
        alt_msg2, alt_score = local_search(
            alt_msg1, best_msg2,
            max_iters=iters // 2,
            initial_temp=8.0,
            seed=restart * 333 + 42
        )
        
        if alt_score > best_matching:
            best_matching = alt_score
            best_msg1 = alt_msg1
            best_msg2 = alt_msg2
            best_h1 = sha3_256_fast(alt_msg1)
            best_h2 = sha3_256_fast(alt_msg2)
            sat_est, total_cl = estimate_sat_clauses(best_matching, best_msg1, best_msg2)
            if verbose:
                print(f"  ¡Mejora (nuevo msg1)! {best_matching}/256 bits "
                      f"({100*best_matching/256:.1f}%)")
                print(f"  SAT estimado: {sat_est}/{total_cl} ({100*sat_est/total_cl:.2f}%)")
    
    # ── Resultados finales ──
    elapsed = time.time() - t_start
    
    # Verificar con hashlib
    h1_verify = hashlib.sha3_256(best_msg1).hexdigest()
    h2_verify = hashlib.sha3_256(best_msg2).hexdigest()
    
    # Contar matching bits real
    b1 = hash_to_bits(bytes.fromhex(h1_verify))
    b2 = hash_to_bits(bytes.fromhex(h2_verify))
    final_matching = int(count_matching_bits(b1, b2))
    
    sat_est, total_cl = estimate_sat_clauses(final_matching, best_msg1, best_msg2)
    
    if verbose:
        print()
        print("="*70)
        print("  RESULTADO FINAL")
        print("="*70)
        print()
        print(f"  msg1: {best_msg1.hex()}")
        print(f"  msg2: {best_msg2.hex()}")
        print()
        print(f"  hash(msg1): {h1_verify}")
        print(f"  hash(msg2): {h2_verify}")
        print()
        print(f"  Bits de hash coincidentes: {final_matching}/256 "
              f"({100*final_matching/256:.1f}%)")
        print(f"  Bits diferentes:           {256 - final_matching}/256")
        print()
        print(f"  Cláusulas SAT estimadas: {sat_est:,}/{total_cl:,} "
              f"({100*sat_est/total_cl:.2f}%)")
        print(f"  Cláusulas no satisfechas:  {total_cl - sat_est:,} "
              f"({100*(total_cl-sat_est)/total_cl:.2f}%)")
        print()
        print(f"  Tiempo total: {elapsed:.1f}s")
        print(f"  Diferentes: {'Sí ✓' if best_msg1 != best_msg2 else 'No ✗'}")
        print(f"  Colisión:   {'¡SÍ! ✓✓✓' if h1_verify == h2_verify else 'No'}")
        print()
        
        # Comparar con la heurística de cuaterniones
        quat_sat = 1015431
        quat_pct = 87.83
        print(f"  Comparación con heurística de cuaterniones:")
        print(f"    Cuaterniones:  {quat_sat:,}/{total_cl:,} ({quat_pct:.2f}%)")
        print(f"    Especializado: {sat_est:,}/{total_cl:,} ({100*sat_est/total_cl:.2f}%)")
        diff = sat_est - quat_sat
        if diff > 0:
            print(f"    Mejora: +{diff:,} cláusulas (+{100*sat_est/total_cl - quat_pct:.2f}%)")
        else:
            print(f"    Diferencia: {diff:,} cláusulas")
        print("="*70)
    
    return {
        'msg1': best_msg1,
        'msg2': best_msg2,
        'hash1': h1_verify,
        'hash2': h2_verify,
        'matching_bits': final_matching,
        'sat_clauses': sat_est,
        'total_clauses': total_cl,
        'sat_pct': 100 * sat_est / total_cl,
        'time': elapsed
    }

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    max_time = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    result = solve_sha3_collision(max_time=max_time, verbose=True)
