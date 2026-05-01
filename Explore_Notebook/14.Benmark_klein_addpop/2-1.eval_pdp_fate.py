import os,sys
import numpy as np
import pandas as pd
import scanpy as sc
import argparse
import pseudodynamics as pdp


os.chdir("/rds/user/wz369/hpc-work/pseudodynamics_plus/")
sys.path.append(os.path.join("/rds/user/wz369/hpc-work/PINN_dynamics", 'scripts'))
import fate_eval_pipeline as fate_eval


p = argparse.ArgumentParser(
    description="Per-cell fate accuracy evaluation for a trained pseudodynamics+ model"
)
p.add_argument("--sim_fn", required=True, help='[ode, sde, sb] the simulation frunc for using velocity and diffusion')
p.add_argument("--config_path", required=True,help="pdp training config : xx.json")
p.add_argument("--celltype_col", default="Annotation")
p.add_argument("--output_path", default=None)
p.add_argument("--t_end_norm", default=0.2, type=float,  help='the normalized integration end time')
p.add_argument("--n_steps", default=200, type=int,  help=' Euler-Maruyama discretisation steps')
p.add_argument("--noise_scale", default=1.0, type=float,  help='Global multiplier on the stochastic term.  Set to 0 for the deterministic limit; sweep [0.1, 0.5, 1.0, 2.0] to tune.')
p.add_argument("--n_sims",       type=int, default=100,
                help="Simulated trajectories per start cell (default 100)")
p.add_argument("--k",            type=int, default=15,
                help="KNN neighbours for cell-type assignment (default 20)")
p.add_argument("--device", default="cuda:0")
args = p.parse_args()


# READ data
adata = sc.read_h5ad("data/klein_addpop.h5ad")
F_obs = pd.read_csv("data/klein/F_obs.csv", index_col=0)
F_obs = F_obs[F_obs.index.astype(str).isin(adata.obs_names)]

config = pdp.ExperimentConfig(args.config_path)
config.from_json(args.config_path, main_dir="/rds/user/wz369/hpc-work/pseudodynamics_plus/")

# get key from config
n_dims = config.dataset_config['n_dimension']
cellstate_key = config.dataset_config.get("cellstate_key", None)
timepoint_key = config.dataset_config.get("timepoint_key", "timepoint_tx_days")

pde_model = pdp.models.pde_params.load_from_checkpoint(
    config.find_lastest_ckpt()
)

# start cells
fobs_idx_str  = F_obs.index.astype(str)
valid_mask  = adata.obs_names.astype(str).isin(fobs_idx_str)
start_cells = adata[valid_mask].obsm[cellstate_key][:,:n_dims]


if args.sim_fn == "ode":
    sim_fn = fate_eval.make_pseudodynamics_sim_fn(t_start_norm=0.0, t_end_norm=args.t_end_norm)
elif args.sim_fn == "sde":
    sim_fn = fate_eval.make_pseudodynamics_sde_sim_fn(t_start_norm=0.0, 
                                                    t_end_norm=args.t_end_norm,
                                                    n_steps = args.n_steps,
                                                    noise_scale = args.noise_scale
                                                    )
elif args.sim_fn == "sb":
    sim_fn = fate_eval.make_pseudodynamics_sb_sim_fn(t_start_norm=0.0, 
                                                    t_end_norm=args.t_end_norm,
                                                    n_steps = args.n_steps,
                                                    noise_scale = args.noise_scale
                                                    )



# prepare data for training label prediction

adata_train = adata[adata.obs.Well != 2]
x_ref = adata_train.obsm[cellstate_key][:, :n_dims].astype(np.float32)
y_ref = adata_train.obs[args.celltype_col].values.astype(str)


result = fate_eval.run_fate_evaluation(start_cells, 
                                        list(adata[valid_mask].obs_names), 
                                        pde_model.to(args.device), 
                                        sim_fn,
                                        F_obs, x_ref, y_ref,
                                        cell_types=None, 
                                        n_sims=100, 
                                        k=20, 
                                        device=args.device)


# ── Save summary CSV ───────────────────────────────────────────────────

output_path = args.output_path 

name = config.experiment_config['save_dir'].split("/")[-2]
output_path = f"results/pseudodynamics+/{name}/t{args.t_end_norm}_n{args.noise_scale}_{args.sim_fn}_fate_eval.csv" if output_path is None else output_path
os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

df_summary = pd.DataFrame([{
    "accuracy":      result["accuracy"],
    "pearson_r":     result["pearson_r"],
    "pearson_p":     result["pearson_p"],
    "n_start_cells": result["n_start_cells"],
    "n_sims":        args.n_sims,
    "k_nn":          args.k,
    "obsm_key":      cellstate_key,
    "n_dims":        n_dims,
    "t_end_norm":    args.t_end_norm,
    "noise_scale":   args.noise_scale,
    "model_dir":     args.config_path,
}])
df_summary.to_csv(output_path, index=False)

print("accuracy",      result["accuracy"])
print("pearson_r",     result["pearson_r"])
print(f"Summary → {output_path}")

# Per-cell-type fate fractions
F_hat = result["F_hat"]
F_obs_al = result["F_obs_aligned"]
fate_tbl = pd.DataFrame({
    "F_obs_mean": F_obs_al.mean(),
    "F_hat_mean": F_hat.mean(),
})
fate_csv = output_path.replace(".csv", f"_celltype_fractions.csv")
fate_tbl.to_csv(fate_csv)
print(f"Cell-type fractions → {fate_csv}")
print(f"\n{fate_tbl.to_string()}")

# Per-cell F_hat
fhat_csv = output_path.replace(".csv", f"_F_hat.csv")
F_hat.to_csv(fhat_csv)
print(f"F_hat per cell → {fhat_csv}")