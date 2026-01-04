import chess
import numpy as np

class QuantumChessBoard:
    def __init__(self, fen=chess.STARTING_FEN):
        self.superpositions = [(chess.Board(fen), 1.0 + 0j)]
        self.normalize()

    def normalize(self):
        amps = np.array([amp for _, amp in self.superpositions])
        norm = np.sqrt(np.sum(np.abs(amps)**2))
        if norm > 0:
            self.superpositions = [(b, a / norm) for b, a in self.superpositions]

    def apply_quantum_move(self, move, split=0.707 + 0j):
        new = []
        for board, amp in self.superpositions:
            copied = board.copy()
            if copied.is_legal(move):
                copied.push(move)
                new.append((copied, amp * split))
            new.append((board, amp * (1 - split)))
        self.superpositions = new
        self.normalize()

    def measure(self, guidance="thriving"):
        probs = [abs(a)**2 for _, a in self.superpositions]
        idx = np.argmax(probs)  # Mercy bias
        collapsed, _ = self.superpositions[idx]
        self.superpositions = [(collapsed, 1.0 + 0j)]
        print(f"Collapsed to thriving: {guidance}")
        return collapsed
