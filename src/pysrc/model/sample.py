from torch import device, LongTensor, no_grad, multinomial, softmax, topk, full, inf

from pysrc.model.pytorch_model import PytorchModel


def melody_token_ids(id2tok: dict[int, str]) -> list[int]:
    """Tokens that may follow the control prefix: melody tokens and EOS."""
    return [i for i, tok in id2tok.items() if tok == "<EOS>" or tok.startswith(("<NOTE_", "<PITCH_", "<REST_"))]


def length_guard(id2tok: dict[int, str], bar_divs: int, bars: int) -> dict:
    """Arguments that make generation follow the BARS control, in the form the
    C++ Museformer.generate takes them (sample_tokens applies the same rule).

    Time advances by a REST's length, and by a NOTE's length once its PITCH
    arrives, so generation never stops between a NOTE and its PITCH. EOS is
    masked until the last bar has started; generation stops once `bars` bars
    have elapsed.
    """
    rest_divs, note_divs = [0] * len(id2tok), [0] * len(id2tok)
    for i, tok in id2tok.items():
        if tok.startswith("<REST_"):
            rest_divs[i] = int(tok[6:-1])
        elif tok.startswith("<NOTE_"):
            note_divs[i] = int(tok[6:-1])
    return {
        "rest_divs": rest_divs,
        "note_divs": note_divs,
        "eos_after_divs": (bars - 1) * bar_divs,
        "stop_at_divs": bars * bar_divs,
    }


def sample_tokens(
        model: PytorchModel, prefix: list[int], id2tok: dict[int, str], system: device,
        max_tokens: int, temperature: float = 1.0, top_k: int = 16,
        bar_divs: int | None = None, bars: int | None = None
) -> list[int]:
    """Continue a control prefix with top-k sampling until EOS or max_tokens.

    With bar_divs and bars, the length follows the BARS control (see
    length_guard). Returns the whole sequence (prefix included, EOS excluded).
    """
    model.eval()

    allowed = full((len(id2tok),), -inf, device=system)
    allowed[melody_token_ids(id2tok)] = 0.0
    no_eos = allowed.clone()
    no_eos[1] = -inf
    guard = length_guard(id2tok, bar_divs, bars) if bars is not None and bar_divs is not None else None

    output = list(prefix)
    elapsed = pending = 0
    with no_grad():
        while len(output) < max_tokens:
            if guard and elapsed >= guard["stop_at_divs"]:
                break
            mask = no_eos if guard and elapsed < guard["eos_after_divs"] else allowed

            cur = LongTensor([output]).to(system)
            logits = model(cur)[0, -1] / temperature + mask

            vals, idxs = topk(logits, top_k)
            next_id = idxs[multinomial(softmax(vals, dim=-1), 1)].item()
            if next_id == 1:
                break
            output.append(next_id)

            if guard:
                if guard["rest_divs"][next_id]:
                    elapsed += guard["rest_divs"][next_id]
                elif guard["note_divs"][next_id]:
                    pending = guard["note_divs"][next_id]
                else:
                    elapsed += pending
                    pending = 0
    return output
