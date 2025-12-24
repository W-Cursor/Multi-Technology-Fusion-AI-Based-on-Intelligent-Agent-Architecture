"""equilibrium.py  独立系统运行用来计算生产过程的平衡指标"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence


DEFAULT_MINERAL_DATA = {
    "bauxite": [50.0, 8.0, 20.0, 3.76, 0.0, 5.0, 0.0, 0.0, 13.24],
    "lime": [0.0, 0.0, 0.0, 0.0, 85.0, 0.0, 0.0, 0.0, 15.0],
    "alumina": [99.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
}

DEFAULT_PROCESS_PARAMS = {
    "B6": 180.0,
    "B7": 15.0,
    "B8": 2.835,
    "B9": 1.27,
    "B10": 340.0,
    "B11": 1.2,
    "B12": 2.265,
    "B13": 1.0,
    "B14": 0.63,
    "B15": 3.0,
    "B16": 145.0,
    "B17": 1.325,
    "B18": 1.317,
    "B19": 1.355,
    "B20": 333.0,
    "B21": 420.0,
    "D6": 550.0,
    "D7": 15.0,
    "D8": 1.0,
    "D9": 5.0,
    "D10": 4.0,
    "D11": 1.0,
    "D12": 15.0,
    "D13": 612.0,
    "D14": 3.5,
    "D15": 1.0,
    "D16": 1.2,
    "D17": 320.0,
    "D18": 80.0,
    "F6": 51.3,
    "F7": 146.0,
    "F8": 19.4,
    "F9": 1.268,
    "F10": 1.5,
    "F11": 1.0,
    "F12": 0.85,
    "F13": 1.1,
    "F14": 0.9,
    "F15": 165.0,
    "F16": 4.7,
    "F17": 10.0,
    "F18": 80.0,
}


def _coerce_sequence(values: Sequence[float | int | str] | None, defaults: Sequence[float]) -> List[float]:
    if not isinstance(values, Sequence):
        return list(defaults)
    coerced: List[float] = []
    for index, default_value in enumerate(defaults):
        try:
            raw = values[index]
        except IndexError:
            coerced.append(float(default_value))
            continue
        try:
            coerced.append(float(raw))
        except (TypeError, ValueError):
            coerced.append(float(default_value))
    return coerced


def _coerce_process_params(payload: Dict[str, float | int | str] | None) -> Dict[str, float]:
    params = DEFAULT_PROCESS_PARAMS.copy()
    if not payload:
        return params
    for key, default_value in DEFAULT_PROCESS_PARAMS.items():
        if key in payload:
            try:
                params[key] = float(payload[key])
            except (TypeError, ValueError):
                params[key] = default_value
    return params


def calculate_equilibrium(
    mineral_data: Dict[str, Iterable[float | int | str]] | None = None,
    process_params: Dict[str, float | int | str] | None = None,
) -> Dict[str, Dict[str, float]]:
    mineral_payload = mineral_data or {}
    bauxite = _coerce_sequence(mineral_payload.get("bauxite"), DEFAULT_MINERAL_DATA["bauxite"])
    lime = _coerce_sequence(mineral_payload.get("lime"), DEFAULT_MINERAL_DATA["lime"])
    alumina = _coerce_sequence(mineral_payload.get("alumina"), DEFAULT_MINERAL_DATA["alumina"])

    params = _coerce_process_params(process_params)

    B2, C2, D2, E2, F2, G2, H2, I2, J2 = bauxite
    B3, C3, D3, E3, F3, G3, H3, I3, J3 = lime
    B4, C4, D4, E4, F4, G4, H4, I4, J4 = alumina

    B6 = params["B6"]
    B7 = params["B7"]
    B8 = params["B8"]
    B9 = params["B9"]
    B10 = params["B10"]
    B11 = params["B11"]
    B12 = params["B12"]
    B13 = params["B13"]
    B14 = params["B14"]
    B15 = params["B15"]
    B16 = params["B16"]
    B17 = params["B17"]
    B18 = params["B18"]
    B19 = params["B19"]
    B20 = params["B20"]
    B21 = params["B21"]
    D6 = params["D6"]
    D7 = params["D7"]
    D8 = params["D8"]
    D9 = params["D9"]
    D10 = params["D10"]
    D11 = params["D11"]
    D12 = params["D12"]
    D13 = params["D13"]
    D14 = params["D14"]
    D15 = params["D15"]
    D16 = params["D16"]
    D17 = params["D17"]
    D18 = params["D18"]
    F6 = params["F6"]
    F7 = params["F7"]
    F8 = params["F8"]
    F9 = params["F9"]
    F10 = params["F10"]
    F11 = params["F11"]
    F12 = params["F12"]
    F13 = params["F13"]
    F14 = params["F14"]
    F15 = params["F15"]
    F16 = params["F16"]
    F17 = params["F17"]
    F18 = params["F18"]

    B24 = ((C2 * 0.01) + (B12 * 0.01 * C3 * 0.01)) * B13 / ((B2 * 0.01) + (B12 * 0.01 * B3 * 0.01)) * 100
    A26 = D8 + F10 + F11 + B24
    B26 = (1 - A26 * 0.01) * 100
    A28 = (1000 * B4 * 0.01) / ((B2 * 0.01 + B12 * 0.01 * B3 * 0.01) * B26 * 0.01)
    B28 = A28 * B12 * 0.01

    B30 = A28 * B2 * 0.01
    C30 = A28 * C2 * 0.01
    D30 = A28 * D2 * 0.01
    E30 = A28 * E2 * 0.01
    F30 = A28 * F2 * 0.01
    G30 = A28 * G2 * 0.01 + A28 * 0.001 * F18
    H30 = A28 * H2 * 0.01
    I30 = A28 * J2 * 0.01

    B32 = B28 * F3 * 0.01
    C32 = B28 * C3 * 0.01
    D32 = B28 * D3 * 0.01
    E32 = B28 * B3 * 0.01
    F32 = B28 * H3 * 0.01
    I32 = B28 * J3 * 0.01

    B33 = B24 * (B30 + E32) * 0.01
    A35 = (C30 + C32) * B14
    B35 = 62 / 44 * (H30 + F32) * F12
    C35 = I4 * 0.01 * 1000
    D35 = A35 + C35 + D7
    E35 = B35 + (D11 * 62 / 44) - D15

    A53 = C30 + C32
    B53 = D30 + D32
    C53 = F30 + B32
    D53 = E30
    E53 = B33
    F53 = A35
    G53 = I30 + I32
    H53 = A53 + B53 + C53 + D53 + E53 + F53 + G53
    I53 = H53 * B15 * 0.01

    F138 = (1 - (E53 / H53 * C2 / (B2 * A53 / H53))) * 100
    F35_val = (
        B30 * (F138 + E32) / 100
        + B14 * (C30 + C32) * 1.645 / B18
        + 1.41 * (H30 + F32) * 1.645 / B18
        + D7 * 1.645 / B18
    ) / (B6 * (1.645 / B18 - 1.645 / B8))
    F35 = F35_val
    G35 = D11 * 62 / 44 * F12
    B37 = B6 * F35
    C37 = B7 * F35
    D37 = 44 / 62 * C37
    E37 = 1.645 * (B37 / B8)
    F37 = F35 * B9 * 1000 - (B37 + C37 + D37 + E37)
    G37 = F35 * B9 * 1000

    A39 = D35
    B39 = A39 / B10
    C39 = B39 * B11 * 1000 - A39
    A41 = E35 / D17
    B41 = A41 * 100 / (D18 * 0.01)
    C41 = B41 * (1 - D18 * 0.01)
    D41 = A41 * D17
    E41 = B35 + G35 + C41 - D15
    F41 = D41 * 1000 / D17
    A43 = F35 + B39 + A41

    L45 = B30 + C30 + D30 + E30 + F30 + G30 + H30 + I30
    L46 = B32 + C32 + D32 + E32 + F32 + I32
    L47 = G37
    L48 = A39 + C39
    L49 = D41 + C41 + F41

    A63 = E41
    B63 = A63 * 124 / 62
    C63 = B63 * 106 / 124
    D63 = B63 - C63
    E63 = C63 * D18 * 0.01
    F63 = E63 * 74 / 106 * D16
    G63 = F63 / 1.32 / F3 * 100
    H63 = G63 * (1 - F3 * 0.01)
    I63 = F63 * 100 / 74
    J63 = I63 + H63
    K63 = G63 * F14 / F15 * 1000 * F13
    L63 = K63 - F63
    L49 = D41 + C41 + F41 + L63

    B50 = B30 + E32 + E37
    C50 = C30 + C32
    D50 = D32 + D30
    E50 = E30
    F50 = B32 + F30
    G50 = B37
    H50 = C37
    I50 = H30 + F32 + D37
    J50 = G30 + F37
    K50 = I30 + I32
    L50 = L45 + L46 + L47

    B57 = B50 - E53
    G57 = B37 - F53
    H57 = C37
    I57 = I50
    J57 = J50 - I53
    L57 = B57 + G57 + H57 + I57 + J57

    A61 = H53
    B61 = A61 + I53
    C61 = B61 * (1000 / B20 - 1)
    E61 = C61 / B19 / 1000 * B16
    D61 = E61 * 1.645 / B17
    F61 = H50 / (G50 - F53) * E61
    G61 = 44 / 62 * F61
    H61 = C61 - (D61 + E61 + F61 + G61)
    I61 = B61 * (1000 / B20)

    A65 = A61 + J63
    E65 = B61 / 1000 * D9
    F65 = E61 * E65 / (E61 + F61)
    G65 = E65 - F65
    H65 = G65 * 44 / 62
    I65 = 1.645 * F65 / B17

    A67 = D61 - I65
    B67 = E61 - F65
    C67 = F61 - G65
    D67 = G61 - H65

    A80 = B50 + A67
    B80 = G50 + B67
    C80 = H50 + C67
    D80 = I50 + D67
    E80 = B80 / B16 * B19 * 1000
    F80 = E80 - (A80 + B80 + C80 + D80)
    G80 = E80 - I61
    H80 = G80 / B19 / 1000
    I80 = F16 * 0.01 * G80
    J80 = G80 + I80
    K80 = I80 / F14 * F15 / 1000 / F13
    L80 = K80 * 1.32 * F3 / 100
    M80 = I80 - L80
    N80 = I80 * F17 * 0.01
    O80 = N80 * 2.5

    A82 = A80 - D61 - E53
    B82 = B80 - E61 - F53
    C82 = C80 - F61
    D82 = D80 - G61
    E82 = G80 - A82 - B82 - C82 - D82

    B65 = A61 + J63 + N80
    C65 = B65 + I53
    K65 = C65 * (1000 / B21)
    L65 = B65 * 28 / 72
    M65 = K65 - L65 - C65

    A106 = B30 + E32 - E53 - I65 - (B30 + E32) * 0.01 * F11 - B4 * 10
    B106 = D7 - F65 - G65
    C106 = B82 / (B82 + C82) * B106
    D106 = B106 - C106

    A108 = (A82 - C106) * F6 * 0.01
    B108 = A108 * 54 / 102
    C108 = A108 + B108
    E108 = E82 * D10 * 0.01

    G108 = L47 - L48 - L49
    H112 = A108 + B108
    A69 = H112 / (1 - D12 * 0.01)
    B69 = A69 - H112
    C69 = B69 / F9 / 1000 * F7 * (1 + D10 * 0.01)
    D69 = B69 / F9 / 1000 * F8 * (1 + D10 * 0.01)
    E69 = C69 / (B8 - 0.1) * 1.645
    F69 = D69 * 44 / 62
    G69 = B69 - (C69 + D69 + E69 + F69)
    I69 = H112 * D14 * 0.01

    A71 = C69 / (B69 + D13) * I69
    B71 = D69 / (B69 + D13) * I69
    C71 = E69 / (B69 + D13) * I69
    D71 = B71 * 44 / 62
    E71 = I69 - (A71 + B71 + C71 + D71)

    E106 = D106 + G65 + B71
    F106 = A106 / A82 * E82
    F108 = E82 - (E108 + B108 + G69 + F106)

    B110 = A82
    C110 = B82
    D110 = C82
    E110 = D82
    F110 = E82 + M80
    G110 = L80
    H110 = B110 + C110 + D110 + E110 + F110 + G110

    B111 = A106
    C111 = C106
    D111 = D106
    E111 = D111 / 1.408
    F111 = F106
    H111 = B111 + C111 + D111 + E111 + F111

    B112 = A108
    F112 = B108
    H112_final = B112 + F112

    B113 = E69
    C113 = C69
    D113 = D69
    E113 = F69
    F113 = G69
    H113 = E69 + C69 + D69 + F69 + G69

    B114 = B113 - C71
    C114 = C113 - A71
    D114 = D113 - B71
    E114 = E113 - D71
    F114 = F113 + D13 - E71
    H114 = B114 + C114 + D114 + E114 + F114

    B115 = B110 - B111 - B112 - B113 - B114
    C115 = C110 - C111 - C113 - C114
    D115 = D110 - D111 - D113 - D114
    E115 = E110 - E111 - E113 - E114
    F115 = F110 - F111 - F112 - F113 - F114
    G115 = G110
    H115 = B115 + C115 + D115 + E115 + F115 + G115

    H108 = H115 + H114 - G108
    G74 = D13

    A125 = (B30 + E32) * F11 * 0.01
    B125 = A108 + C71 - A125

    F136 = 1.645 * B6 * (B8 - B18) / (B8 * B18)
    F137 = (1 - 1 / (B2 / C2)) * 100
    F139 = F138 / F137 * 100 if F137 else 0.0
    F140 = 0.608 / (B2 / C2 - 1) * 1000 if (B2 / C2 - 1) else 0.0
    F142 = (1 - ((C110 + D110) / B110) / ((B37 + C37) / E37)) * 100
    F143 = L50 / 1000

    H82 = E80 - L50 + D6
    J82 = H82 + K65 - I61 - O80 - L63

    return {
        "materials": {
            "A28": A28,
            "B28": B28,
            "L48": L48,
            "L49": L49,
            "L47": L47,
            "L50": L50,
            "L57": L57,
            "D6": D6,
            "A61": A61,
            "B65": B65,
            "G80": G80,
            "J80": J80,
            "J82": J82,
            "H82": H82,
            "C108": C108,
            "H115": H115,
            "G74": G74,
            "H108": H108,
            "E80": E80,
            "wash_water_total": K63 + I80,
        },
        "efficiency": {
            "F136": F136,
            "F137": F137,
            "F138": F138,
            "F139": F139,
            "F140": F140,
            "F142": F142,
            "F143": F143,
        },
        "intermediate": {
            "B24": B24,
            "A26": A26,
            "B26": B26,
            "A35": A35,
            "B35": B35,
            "C35": C35,
            "D35": D35,
            "E35": E35,
            "H53": H53,
            "I53": I53,
            "A63": A63,
            "B63": B63,
            "E61": E61,
            "I80": I80,
            "K80": K80,
            "A108": A108,
            "B108": B108,
            "C108": C108,
        },
    }
