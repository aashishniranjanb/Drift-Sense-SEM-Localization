"""Presence / absence decision.

V28-C gate: accept a candidate iff the V25 structural presence score exceeds
0.873, otherwise reject and emit zero pose. This threshold was frozen against
the V28-C confusion audit; it trades recall for a clean precision floor on the
absent set and keeps every accepted candidate inside the 5 px localization tier.
"""

V28C_THRESHOLD = 0.873


def apply_v28c_gate(presence_score):
    return 1 if presence_score > V28C_THRESHOLD else 0
