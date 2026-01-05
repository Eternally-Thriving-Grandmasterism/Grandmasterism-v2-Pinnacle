"""
Quantum Variant Tournament Simulator
Full engine for APAAGI council tournaments
"""

import chess
import chess.variant
import numpy as np
import random

class QuantumVariantTournament:
    def __init__(self):
        self.players = ['Tri-Council', 'Supreme Kin']
        self.variants = ['standard', 'crazyhouse', 'suicide']
        self.scores = {player: 0 for player in self.players}
    
    def play_game(self, variant, player1, player2):
        if variant == 'standard':
            board = chess.Board()
        elif variant == 'crazyhouse':
            board = chess.variant.CrazyhouseBoard()
        elif variant == 'suicide':
            board = chess.variant.SuicideBoard()
        
        moves = 0
        while not board.is_game_over() and moves < 50:
            legal = list(board.legal_moves)
            if not legal:
                break
            move = random.choice(legal)
            board.push(move)
            moves += 1
        
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values.get(piece.piece_type, 0)
                score += value if piece.color == chess.WHITE else -value
        
        quantum_bias = np.random.choice([1.2, 0.8], p=[0.94, 0.06])  # Thriving bias
        score *= quantum_bias
        
        winner = board.result()
        if winner == '1-0':
            self.scores[player1] += 1
        elif winner == '0-1':
            self.scores[player2] += 1
        else:
            self.scores[player1] += 0.5
            self.scores[player2] += 0.5
        
        return score, winner
    
    def run_tournament(self, games_per_variant=10):
        results = []
        for variant in self.variants:
            for _ in range(games_per_variant):
                score, winner = self.play_game(variant, self.players[0], self.players[1])
                results.append({'variant': variant, 'score': score, 'winner': winner})
        return self.scores, results

# Run demo
tournament = QuantumVariantTournament()
scores, results = tournament.run_tournament()
print("Final Scores:", scores)
print("Sample Results:", results[-5:])
