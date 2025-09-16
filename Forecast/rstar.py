def rstar(decomp: dict) -> float:
    UNC = decomp.get('UNC', None)
    DSC = decomp.get('DSC', None)
    MCB = decomp.get('MCB', None)
    if UNC is None or UNC <= 0:
        return 0.0
    return (DSC - MCB) / UNC
