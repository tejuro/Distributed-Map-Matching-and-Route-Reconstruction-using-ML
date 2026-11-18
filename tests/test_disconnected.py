
from hmm_viterbi.viterbi import viterbi_decode_with_restarts

emissions = [
    {10: 0.9},
    {20: 0.9},
    {30: 0.9}
]

def transition(previous_arc, current_arc, index):
    # Only allow the transition from 20 to 30
    if (previous_arc, current_arc) == (20, 30):
        return 0.8

    return 0.0


path, restarts = viterbi_decode_with_restarts(
    emissions,
    transition
)

print("Predicted path:", path)
print("Restart indices:", restarts)
