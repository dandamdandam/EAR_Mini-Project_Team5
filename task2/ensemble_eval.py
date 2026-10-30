import json

import numpy as np
import torch

from model import QueueNet
from ode import DEVICE
from tau_select import (add_cond_feature, bootstrap_tau, build_pools,
                        collect_entries, percond_eval, score_matrix)
from utils import OUT_DIR

PAIRS = [('cond_s123', 'noode_s123'), ('cond_s124', 'noode_s124'), ('cond_s125', 'noode_s125')]


def load(name):
    sd = torch.load(f'outputs/sweep/{name}.pt', weights_only=False)['model']
    c_in = int(sd['proj.0.weight'].shape[1])
    m = QueueNet(c_in=c_in, no_ode=c_in in (18, 16)).to(DEVICE)
    m.load_state_dict(sd)
    return m, c_in in (21, 18)


def avg_entries(ea, eb):
    return [{'panel': x['panel'], 'probs': (x['probs'] + y['probs']) / 2,
             'qt': x['qt'], 'cond': x['cond']} for x, y in zip(ea, eb)]


def main():
    pools_ode = build_pools(np.random.default_rng(0), splits=('val', 'test'))
    add_cond_feature(pools_ode)
    pools_clim = build_pools(np.random.default_rng(0), splits=('val', 'test'), no_ode=True)
    add_cond_feature(pools_clim)

    rows = []
    for a, b in PAIRS:
        ma, ua = load(a)
        mb, ub = load(b)
        S = {}
        for split in ('val', 'test'):
            ents = avg_entries(collect_entries(ma, pools_ode[split], ua),
                               collect_entries(mb, pools_clim[split], ub))
            S[split] = score_matrix(ents)
        Sv, cv = S['val']
        St, ct = S['test']
        taus = {c: bootstrap_tau(Sv[cv == c])[0] for c in (0, 1)}
        val_eq, val_per = percond_eval(Sv, cv, taus)
        test_eq, test_per = percond_eval(St, ct, taus)
        rows.append({'pair': f'{a}+{b}', 'tau_onset': taus[0], 'tau_ongoing': taus[1],
                     'val_eq': val_eq, 'test_eq': test_eq,
                     'test_onset': test_per[0], 'test_ongoing': test_per[1]})
        print(f"{rows[-1]['pair']}: val_eq={val_eq:.4f} test_eq={test_eq:.4f} "
              f"onset={test_per[0]:.4f} ongoing={test_per[1]:.4f} "
              f"tau={taus[0]:.3f}/{taus[1]:.3f}", flush=True)

    (OUT_DIR / 'sweep' / 'ensemble_report.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
