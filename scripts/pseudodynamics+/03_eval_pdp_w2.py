import os, sys
import numpy as np
import pandas as pd
import scanpy as sc
import argparse
import pseudodynamics as pdp

os.chdir("/rds/user/wz369/hpc-work/pseudodynamics_plus")
sys.path.append(os.path.join("/rds/user/wz369/hpc-work/pseudodynamics_plus", 'scripts'))
import fate_eval_pipeline as fate_eval


p = argparse.ArgumentParser(
    description="W2 distance evaluation for a trained pseudodynamics+ model (Klein dataset)"
)
p.add_argument("--sim_fn", required=True, choices=["ode", "sde", "sb"],
               help='[ode, sde, sb] the simulation func for using velocity and diffusion')
p.add_argument("--config_path", required=True, help="pdp training config : xx.json")
p.add_argument("--output_path", default=None)
p.add_argument("--t_end_norm", default=0.2, type=float, help='the normalized integration end time')
p.add_argument("--n_steps", default=200, type=int, help='Euler-Maruyama discretisation steps')
p.add_argument("--noise_scale", default=1.0, type=float,
               help='Global multiplier on the stochastic term.')
p.add_argument("--n_sims", type=int, default=10,
               help="Simulated trajectories per start cell (default 10)")
p.add_argument("--device", default="cuda:0")
p.add_argument("--skip_per_clone", action="store_true",
               help="Skip the 92-clone klein_w2_v2 step (population W2 only).")
args = p.parse_args()


# READ data
adata = sc.read_h5ad("data/klein_addpop.h5ad")
clone_proportions = pd.read_csv("data/klein/clone_proportions.csv", index_col=0)

config = pdp.ExperimentConfig(args.config_path)
config.from_json(args.config_path, main_dir='/rds/user/wz369/hpc-work/pseudodynamics_plus/')
# ── Compute scaler for unstandardized W2 ──
cellstate_key = config.dataset_config.get("cellstate_key", None)
n_dims = config.dataset_config['n_dimension']


if "pca" in cellstate_key:
    scaler = adata.uns['PC_scaler']
elif "DM" in cellstate_key:
    scaler = adata.uns['DM_scaler']
else:
    scaler = None

# build sim_fn
if args.sim_fn == "ode":
    sim_fn = fate_eval.make_pseudodynamics_sim_fn(t_start_norm=2.0, t_end_norm=args.t_end_norm)
elif args.sim_fn == "sde":
    sim_fn = fate_eval.make_pseudodynamics_sde_sim_fn(t_start_norm=2.0,
                                                      t_end_norm=args.t_end_norm,
                                                      n_steps=args.n_steps,
                                                      noise_scale=args.noise_scale)
elif args.sim_fn == "sb":
    sim_fn = fate_eval.make_pseudodynamics_sb_sim_fn(t_start_norm=2.0,
                                                     t_end_norm=args.t_end_norm,
                                                     n_steps=args.n_steps,
                                                     noise_scale=args.noise_scale)

# test set
test_ad = adata[adata.obs.Well == 2]

w2_pop = fate_eval.klein_w2_v1(test_ad, clone_proportions, config,
                               args.device, sim_fn,
                               n_sims=args.n_sims,
                               scaler=scaler)

print(f"W2 scaled: {w2_pop['w2_scaled']:.6f}")
print(f"W2 raw:    {w2_pop['w2_raw']:.6f}")

# ── Save summary CSV ───────────────────────────────────────────────────
output_path = args.output_path

parts = config.experiment_config['save_dir'].rstrip('/').split('/')
if 'logs' in parts:
    logs_idx = parts.index('logs')
    name_parts = parts[logs_idx + 1:-1]
    name = "__".join(name_parts) if name_parts else parts[-2]
else:
    name = parts[-2]
output_path = (f"results/pseudodynamics+/{name}/t{args.t_end_norm}_n{args.noise_scale}_{args.sim_fn}_w2_eval.csv"
               if output_path is None else output_path)
os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

df_summary = pd.DataFrame([{
    "w2_scaled":     w2_pop["w2_scaled"],
    "w2_raw":        w2_pop["w2_raw"],
    "n_sims":        args.n_sims,
    "t_end_norm":    args.t_end_norm,
    "noise_scale":   args.noise_scale,
    "sim_fn":        args.sim_fn,
    "n_steps":       args.n_steps,
    "model_dir":     args.config_path,
    "unstandardized": scaler is not None,
}])
df_summary.to_csv(output_path, index=False)
print(f"Summary → {output_path}")

# ── Per-clone W2 (v2) ────────────────────────────────────────────────
if args.skip_per_clone:
    print("Skipping per-clone W2 (--skip_per_clone).")
    sys.exit(0)

df_clone_w2 = fate_eval.klein_w2_v2(test_ad, clone_proportions, config,
                                    args.device, sim_fn,
                                    n_sims=args.n_sims,
                                    scaler=scaler)

clone_w2_csv = output_path.replace(".csv", "_per_clone.csv")
df_clone_w2.to_csv(clone_w2_csv, index=False)
print(f"Per-clone W2 → {clone_w2_csv}")
print(f"Mean clone W2 raw:    {df_clone_w2['w2_raw'].mean():.6f}")
print(f"Mean clone W2 scaled: {df_clone_w2['w2_scaled'].mean():.6f}  (n_clones={len(df_clone_w2)})")
print(f"\n{df_clone_w2.to_string(index=False)}")
