"""
Stockfish Variant Support — Standard/Chess960 Native + Fallback for Others
Fairy-Stockfish recommended for full variants (lczero.org fork)
"""

import chess
import chess.variant
import chess.engine

class StockfishVariantSupport:
    def __init__(self, stockfish_path: str = "stockfish", fairy_path: str = "fairy-stockfish"):
        self.stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.stockfish.configure({"UCI_Variant": "chess960"})  # Chess960 ready
        self.fairy = chess.engine.SimpleEngine.popen_uci(fairy_path) if fairy_path else None
        self.fallback_minimax = self.simple_minimax_fallback

    def evaluate_variant(self, fen: str, variant: str = "standard", depth: int = 20) -> dict:
        """Eval for variant — Stockfish for standard/960, Fairy for others, fallback minimax"""
        try:
            if variant in ["standard", "chess960"]:
                board = chess.Board(fen) if variant == "standard" else chess.Board(fen, chess960=True)
                info = self.stockfish.analyse(board, chess.engine.Limit(depth=depth))
            elif self.fairy:
                self.fairy.configure({"UCI_Variant": variant})
                board = getattr(chess.variant, variant.title() + "Board")(fen)
                info = self.fairy.analyse(board, chess.engine.Limit(depth=depth))
            else:
                board = getattr(chess.variant, variant.title() + "Board", chess.Board)(fen)
                score = self.fallback_minimax(board)
                return {"score_cp": score, "best_move": "fallback", "variant": variant, "note": "Fairy-Stockfish recommended"}
            
            score = info["score"].white().score(mate_score=10000)
            best_move = info["pv"][0] if "pv" in info else None
            return {
                "fen": fen,
                "variant": variant,
                "score_cp": score,
                "best_move": board.san(best_move) if best_move else "None",
                "thriving_eval": "equitable_abundance" if abs(score) < 100 else "mercy_adjusted"
            }
        except Exception as e:
            return {"error": str(e), "fallback": self.fallback_minimax(chess.Board(fen)), "recommend": "Install Fairy-Stockfish"}

    def simple_minimax_fallback(self, board: chess.Board) -> int:
        """Fallback minimax for unsupported variants"""
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                score += value if piece.color == chess.WHITE else -value
        return score

    def quit(self):
        self.stockfish.quit()
        if self.fairy:
            self.fairy.quit()

if __name__ == "__main__":
    stock = StockfishVariantSupport()
    print(stock.evaluate_variant(chess.STARTING_FEN, "standard"))
    print(stock.evaluate_variant(chess.variant.CrazyhouseBoard().fen(), "crazyhouse"))
