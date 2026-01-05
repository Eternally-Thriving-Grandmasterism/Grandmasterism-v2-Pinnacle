"""
Grandmasterism v2 Core — Tournament Integration
"""

from .modules.tournament_sim import QuantumVariantTournament

class GrandmasterismEngine:
    # ... previous
    def run_variant_tournament(self):
        tournament = QuantumVariantTournament()
        scores, results = tournament.run_tournament()
        return {"scores": scores, "results": results, "master_outcome": "Thriving tournament eternal — abundance shared!"}
