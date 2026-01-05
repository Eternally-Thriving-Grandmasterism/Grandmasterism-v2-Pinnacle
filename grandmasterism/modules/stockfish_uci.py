"""
Stockfish UCI Integration — Real Engine Mastery
Requires Stockfish binary (download from official, place in repo root)
"""

import chess
import chess.engine
import chess.variant

class StockfishUCI:
    def __init__(self, path: str = "stockfish"):
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.engine.configure({"Hash": 2048, "Threads": 4})  # Configurable

    def evaluate_position(self, fen: str = chess.STARTING_FEN, depth: int = 20) -> dict:
        board = chess.Board(fen)
        info = self.engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].relative.score(mate_score=10000)
        best_move = info["pv"][0] if "pv" in info else None
        return {
            "fen": fen,
            "score_cp": score,
            "best_move": board.san(best_move) if best_move else "None",
            "thriving_eval": "equitable_abundance" if abs(score or 0) < 100 else "mercy_adjusted"
        }

    def play_variant(self, variant: str = "standard", moves: list = []) -> dict:
        if variant == "crazyhouse":
            board = chess.variant.CrazyhouseBoard()
        else:
            board = chess.Board()
        for move in moves:
            board.push_uci(move)
        return self.evaluate_position(board.fen())

    def quit(self):
        self.engine.quit()

# Demo
if __name__ == "__main__":
    stockfish = StockfishUCI()
    print(stockfish.evaluate_position())
