import SIREN.network, SIREN.application, SIREN.utils

REGISTRY = {}

REGISTRY['Interplane'] = SIREN.application.InterplanePrediction
REGISTRY['Motion'] = SIREN.application.Motion_Correction