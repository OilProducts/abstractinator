class AttnCache:
    """Simple container so we don’t juggle nested dicts."""

    def __init__(self):
        self.self_k = self.self_v = None  # self-attn keys/vals
        self.cross = {}  # layer_name → {k,v,seg_ptr}
