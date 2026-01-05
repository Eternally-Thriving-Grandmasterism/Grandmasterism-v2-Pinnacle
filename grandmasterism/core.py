"""
Grandmasterism v2 Core — Full Variants + UCI + AlphaZero + Quantum Expanded
"""

# ... previous imports
from .modules.stockfish_uci import StockfishUCI
from .modules.alphazero_variant import AlphaZeroProxy, mcts_simulation
from .modules.quantum_chess.quantum_moves import apply_entangled_move, quantum_pawn_promotion

class GrandmasterismEngine:
    def __init__(self):
        # ... previous
        self.stockfish = StockfishUCI()
        self.alphazero = AlphaZeroProxy()

    def stockfish_eval(self, fen: str = chess.STARTING_FEN):
        return self.stockfish.evaluate_position(fen)

    def alphazero_move(self, board: chess.Board):
        return mcts_simulation(board, self.alphazero)

    def entangled_quantum(self, move1: str, move2: str):
        uci1 = chess.Move.from_uci(move1)
        uci2 = chess.Move.from_uci(move2)
        apply_entangled_move(self.quantum_board, uci1, uci2)
        return self.quantum_board.measure("entangled_thriving")
