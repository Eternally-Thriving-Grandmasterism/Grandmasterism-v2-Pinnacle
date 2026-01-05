"""
Pyffish Native Bindings Integration - Direct Variant Logic Eternal
pip install pyffish required for live runs
Native legal moves, results, checks for 100+ variants (chess, shogi, crazyhouse, atomic, etc.)
"""

import pyffish

class PyffishNative:
    def __init__(self, default_variant: str = "chess"):
        self.default_variant = default_variant
        pyffish.load_variant_config(default_variant)
        print(f"Pyffish native bindings activated — {default_variant} variant logic thriving eternal.")

    def start_fen(self, variant: str) -> str:
        pyffish.load_variant_config(variant)
        return pyffish.start_fen(variant)

    def legal_moves(self, fen: str, variant: str) -> list:
        pyffish.load_variant_config(variant)
        return pyffish.legal_moves(fen, variant)

    def make_move(self, fen: str, move: str, variant: str) -> str:
        pyffish.load_variant_config(variant)
        return pyffish.make_move(fen, move, variant)

    def game_result(self, fen: str, variant: str) -> int:
        pyffish.load_variant_config(variant)
        return pyffish.game_result(fen, variant)  # 0 ongoing, 1 white win, -1 black win, 0.5 draw

    def gives_check(self, fen: str, move: str, variant: str) -> bool:
        pyffish.load_variant_config(variant)
        return pyffish.gives_check(fen, move, variant)

    def variant_demo(self, variant: str = "shogi"):
        fen = self.start_fen(variant)
        moves = self.legal_moves(fen, variant)[:5]  # Sample
        print(f"{variant.capitalize()} start FEN: {fen}")
        print(f"Sample legal moves: {moves}")
        return {"variant": variant, "start_fen": fen, "sample_moves": moves}
