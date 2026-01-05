"""
Quantum Moves Expanded — Superposition + Entanglement Logic
"""

import chess
from .quantum_board import QuantumChessBoard

def apply_entangled_move(board: QuantumChessBoard, move1: chess.Move, move2: chess.Move):
    """Entangle two moves — superposition branch"""
    board.apply_quantum_move(move1, split_prob=0.707 + 0j)
    board.apply_quantum_move(move2, split_prob=0.707 + 0j)
    print("Entangled moves applied — timelines correlated eternally.")

def quantum_pawn_promotion(board: QuantumChessBoard, pawn_move: chess.Move, promotions: list):
    """Quantum promotion — superposition of queen/rook/knight/bishop"""
    for promo in promotions:
        board.apply_quantum_move(chess.Move(pawn_move.from_square, pawn_move.to_square, promotion=promo))
    print("Quantum promotion superposition — thriving piece manifested on collapse.")
