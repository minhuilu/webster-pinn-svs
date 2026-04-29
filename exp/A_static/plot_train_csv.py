import argparse, pandas as pd, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    fig, axs = plt.subplots(3, 2, figsize=(10,9))

    # 1) window
    axs[0,0].plot(df["epoch"], df["L_mstft"], label="L_mstft")
    axs[0,0].plot(df["epoch"], df["L_env"],   label="L_env")
    axs[0,0].plot(df["epoch"], df["L_mstft_full"], label="L_mstft_full")
    axs[0,0].set_title("Audio losses (window/global)"); axs[0,0].legend()

    # 2) PDE and boundaries
    axs[0,1].plot(df["epoch"], df["L_pde"], label="L_pde")
    axs[0,1].plot(df["epoch"], df["L_rad"], label="L_rad")
    axs[0,1].plot(df["epoch"], df["L_smh"], label="L_smh(A_xx)")
    axs[0,1].set_yscale("log"); axs[0,1].legend(); axs[0,1].set_title("Physics losses")

    # 3) total loss and selected weighted contributions
    axs[1,0].plot(df["epoch"], df["L"], label="L(total)")
    axs[1,0].plot(df["epoch"], df["c_mstft"], label="c_mstft")
    axs[1,0].plot(df["epoch"], df["c_pde"],   label="c_pde")
    axs[1,0].plot(df["epoch"], df["c_mfull"], label="c_mstft_full")
    axs[1,0].set_title("Total & contributions"); axs[1,0].legend()

    # 4) scalar
    axs[1,1].plot(df["epoch"], df["p_gain"], label="p_gain")
    axs[1,1].plot(df["epoch"], 1e3*df["tau_sec"], label="tau(ms)")
    axs[1,1].plot(df["epoch"], df["zeta"], label="zeta")
    axs[1,1].legend(); axs[1,1].set_title("Learnable scalars")

    # 5) fsmooth EMA formant weight
    axs[2,0].plot(df["epoch"], df["fsmooth_ema"], label="fsmooth_ema")
    axs[2,0].plot(df["epoch"], df["w_form"], label="w_form")
    axs[2,0].plot(df["epoch"], df["w_form_smooth"], label="w_form_smooth")
    axs[2,0].legend(); axs[2,0].set_title("Formant schedule")

    # 6) learning rate
    axs[2,1].plot(df["epoch"], df["lr_main"], label="lr_main")
    axs[2,1].plot(df["epoch"], df["lr_p_gain"], label="lr_p_gain")
    axs[2,1].plot(df["epoch"], df["lr_tau"], label="lr_tau")
    axs[2,1].plot(df["epoch"], df["lr_zeta"], label="lr_zeta")
    axs[2,1].legend(); axs[2,1].set_title("Learning rates")

    for ax in axs.ravel(): ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(args.out, dpi=150)

if __name__ == "__main__":
    main()
