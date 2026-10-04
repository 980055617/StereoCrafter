"""more_20261004 / temporal lane: window geometry of inpainting_inference.main(), shared by the hook and the scorer.

window_schedule() replicates the loop of inpainting_inference.main (tracked, unchanged) line by line:
    for i in range(0, num_frames, frames_chunk - overlap):
        if i + overlap >= N: break
        if generated is not None and i + frames_chunk > N:        # generated is not None <=> not the first window
            cur_i = max(N + overlap - frames_chunk, 0); cur_overlap = i - cur_i + overlap
        else:
            cur_i = i; cur_overlap = overlap
        input_frames_i = frames_warped[cur_i : cur_i + frames_chunk]
        ... generated = decoded window; if i != 0: generated = generated[cur_overlap:]
=> window k contributes absolute frames [keep_from, keep_to) = [cur_i + (cur_overlap if k else 0), cur_i + nf).
"""


def window_schedule(N, chunk, overlap):
    out = []
    for i in range(0, N, chunk - overlap):
        if i + overlap >= N:
            break
        if out and i + chunk > N:
            cur_i = max(N + overlap - chunk, 0)
            cur_ov = i - cur_i + overlap
        else:
            cur_i = i
            cur_ov = overlap
        nf = min(cur_i + chunk, N) - cur_i
        keep_from = cur_i + (cur_ov if out else 0)
        out.append(dict(i=i, cur_i=cur_i, cur_overlap=cur_ov, nf=nf, keep_from=keep_from, keep_to=cur_i + nf))
    # kept ranges must tile [0, N) exactly
    assert out[0]["keep_from"] == 0 and out[-1]["keep_to"] == N, out
    for a, b in zip(out[:-1], out[1:]):
        assert a["keep_to"] == b["keep_from"], (a, b)
    return out


def classify_transitions(N, chunk, overlap, dcs, n_tr):
    """class of transition t -> t+1 (t < n_tr <= N-1): 'seam' (frames from different windows), 'bnd' (same window,
    different VAE decode chunk, window-LOCAL position), 'in' (same window, same decode chunk)."""
    W = window_schedule(N, chunk, overlap)
    owner = [None] * N
    for k, w in enumerate(W):
        for f in range(w["keep_from"], w["keep_to"]):
            owner[f] = (k, f - w["cur_i"])
    assert all(o is not None for o in owner)
    cls = []
    for t in range(n_tr):
        (k0, p0), (k1, p1) = owner[t], owner[t + 1]
        if k0 != k1:
            cls.append("seam")
        elif p0 // dcs != p1 // dcs:
            cls.append("bnd")
        else:
            cls.append("in")
    return W, cls


if __name__ == "__main__":
    # self-test: overlap 3 seams == the tracked scorer's t = 11k+2 (k >= 1) for the clip lengths in use
    for N in (150, 151, 153, 120, 14, 25):
        for n_tr in (N - 1, N - 4):
            W, cls = classify_transitions(N, 14, 3, 2, n_tr)
            own = [t for t, c in enumerate(cls) if c == "seam"]
            ref = [t for t in range(n_tr) if (t - 2) % 11 == 0 and t >= 13]
            assert own == ref, (N, n_tr, own, ref)
    for N in (150, 151, 153):
        for ov in (3, 5, 7):
            W = window_schedule(N, 14, ov)
            print(f"N={N} ov={ov} windows={len(W)} lengths={[w['nf'] for w in W]} "
                  f"last={W[-1]} seams={[w['keep_from'] - 1 for w in W[1:]]}")
    print("tlib self-test OK")
