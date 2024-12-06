#!/usr/bin/env python

import matplotlib.pyplot as plt
import mplhep
import numpy as np
import os
import uproot

from topsf.config.run2.analysis_sf import config_2017


def hist_ratio_values_errors(h1, h2, sum_errors=False):
    vals = h1.values() / h2.values()
    if sum_errors:
        errs = np.sqrt(
            h1.variances() / h1.values()**2 +
            h2.variances() / h2.values()**2
        ) * vals
    else:
        errs = np.sqrt(h1.variances()) / h2.values()

    return vals, errs


def hist_rdiff_values_errors(h1, h2, sum_errors=False):
    vals = (h1.values() - h2.values()) / h2.values()
    if sum_errors:
        errs = np.sqrt(
            h1.variances() / h1.values()**2 +
            h2.variances() / h2.values()**2
        ) * vals
    else:
        errs = np.sqrt(h1.variances()) / h2.values()

    return vals, errs


def plot_comparison(hists, labels, output_filename, annotate_cfgs=None, style_cfgs=None, ratio=True):

    assert len(hists) == len(labels) == 2
    annotate_cfgs = annotate_cfgs or []
    style_cfgs = style_cfgs or []

    # plot the efficiencies
    mplhep.style.use("CMS")
    if ratio:
        fig, axs = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05), sharex=True)
        (ax, rax) = axs
    else:
        fig = plt.figure()
        ax = plt.gca()
        rax = None

    def draw_hist(ax, h, vals, yerr, color, linestyle, linewidth, style_cfg):
        # do plotting using mplhep
        mplhep.histplot(
            vals,
            # bins=h.axes[0].edges(),
            bins=h.axes[0].edges,
            histtype="step",
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            ax=ax,
            **style_cfg,
        )
        ax.bar(
            label=f"{label} - stat. unc.",
            # x=h.axes[0].centers(),
            # width=h.axes[0].edges()[1:] - h.axes[0].edges()[:-1],
            x=h.axes[0].centers,
            width=h.axes[0].edges[1:] - h.axes[0].edges[:-1],
            bottom=vals - yerr,
            height=2 * yerr,
            hatch=hatch,
            facecolor="none",
            linewidth=0,
            color=color,
            edgecolor=color,
            alpha=1.0,
            **style_cfg,
        )

    for i, (h, label, hatch) in enumerate(
        zip(hists, labels, ["///", "\\\\\\"])
    ):
        style_cfg = dict(style_cfgs[i]) if len(style_cfgs) > i else {}

        color = style_cfg.pop("color", f"C{i}")
        linestyle = style_cfg.pop("linestyle", "solid")
        linewidth = style_cfg.pop("linewidth", None)

        vals = h.values()
        # if "CF" in label:
        #     # multiply vals by 5 to correct for missing lumi
        #     vals *= 5
        yerr = np.sqrt(h.variances())
        draw_hist(ax, h, vals, yerr, color, linestyle, linewidth, style_cfg)

        if ratio:
            # rvals, ryerr = hist_ratio_values_errors(h, hists[0], sum_errors=False)
            rvals, ryerr = hist_rdiff_values_errors(h, hists[0], sum_errors=False)
            draw_hist(rax, h, rvals, ryerr, color, linestyle, linewidth, style_cfg)

    # set labels
    mplhep.cms.label(ax=ax, fontsize=22, llabel="Simulation Private Work", loc=0, com="13")
    ax.set_ylabel("Event yield")
    if not ratio:
        ax.set_xlabel("$m_{SD}$ / GeV")
    ax.set_yscale("log")

    # rax.set_ylabel(f"Rel. diff. {labels[0]}")
    if ratio:
        rax.set_ylabel("$\Delta$y / y")
        rax.set_xlabel("$m_{SD}$ / GeV")
        # rax.set_yscale("symlog")
        rax.set_ylim((-3, 3))
        rax.grid(axis="y")
        rax.axhline(0, color="black", linestyle="--")

    # draw legend
    ax.legend(ncol=1, loc="upper right")

    for ann in annotate_cfgs:
        ax.annotate(**ann)

    # save plot
    plt.savefig(f"{output_filename}.pdf")
    plt.savefig(f"{output_filename}.png")


paths = {
    "uhh2": "/nfs/dust/cms/user/matthiej/topsf/work/compare_shapes_cmatthies/ref_templates_very_loose.root",
    "cf": [
        "/nfs/dust/cms/user/matthiej/topsf/data/topsf_store/analysis_run2_sf/topsf.CreateDatacards/run2_sf_2017_nano_v9/calib__default/sel__default/prod__weights__features/inf__uhh2/wp__very_loose/240328_v1/shapes__cat_16_3df8351d8a__var_probejet_msoftdrop_widebins.root",  # noqa
        # "/nfs/dust/cms/user/matthiej/topsf/data/topsf_store/analysis_run2_sf/topsf.CreateDatacards/run2_sf_2017_nano_v9/calib__default/sel__default/prod__weights__features/inf__uhh2/wp__very_tight/240328_v1/shapes__cat_16_6ad4815f64__var_probejet_msoftdrop_widebins.root",  # noqa
        # "/nfs/dust/cms/user/matthiej/topsf/data/topsf_store/analysis_run3_sf/topsf.CreateDatacards/run3_sf_2022_preEE_nano_v12/calib__default/sel__default/prod__weights__features/inf__uhh2/wp__very_tight/241011_v9/shapes__cat_16_6ad4815f64__var_probejet_msoftdrop_widebins.root",  # noqa
        # "/nfs/dust/cms/user/matthiej/topsf/data/topsf_store/analysis_run3_sf/topsf.CreateDatacards/run3_sf_2022_preEE_nano_v12/calib__default/sel__default/prod__weights__features/inf__uhh2/wp__very_loose/240730_v5/shapes__cat_16_3df8351d8a__var_probejet_msoftdrop_widebins.root",  # noqa
    ],
}

suffix = "__UL17__pt_300to400"

uhh2_map = {
    "process": {
        "tt_3q": f"TTbar__MSc_FullyMerged{suffix}",
        "tt_2q": f"TTbar__MSc_SemiMerged{suffix}",
        "tt_0o1q": f"TTbar__MSc_NotMerged{suffix}",
        "tt_bkg": "TTbar__MSc_Background",
        "st_3q": f"ST__MSc_FullyMerged{suffix}",
        "st_2q": f"ST__MSc_SemiMerged{suffix}",
        "st_0o1q": f"ST__MSc_NotMerged{suffix}",
        "st_bkg": "ST__MSc_Background",
        "mj": "QCD",
        "vx": "VJetsAndVV",
    },
    "channel": {
        "1m": "muo",
        "1e": "ele",
    },
    "pass_fail": {
        "pass": "Pass",
        "fail": "Fail",
    },
}

channels = list(uhh2_map["channel"])
processes = list(uhh2_map["process"])
pass_fails = list(uhh2_map["pass_fail"])

compound_processes = {}
for p in ("tt", "st"):
    compound_processes[p] = [
        f"{p}_{m}" for m in ("3q", "2q", "0o1q", "bkg")
    ]

# processes = [
#    ("tt_3q", ["tt_3q"], [f"TTbar__MSc_FullyMerged{suffix}"]),
#    ("tt_2q", ["tt_2q"], [f"TTbar__MSc_SemiMerged{suffix}"]),
#    ("tt_0o1q", ["tt_0o1q"], [f"TTbar__MSc_NotMerged{suffix}"]),
#    ("tt_bkg", ["tt_bkg"], ["TTbar__MSc_Background"]),
#    ("st_3q", ["st_3q"], [f"ST__MSc_FullyMerged{suffix}"]),
#    ("st_2q", ["st_2q"], [f"ST__MSc_SemiMerged{suffix}"]),
#    ("st_0o1q", ["st_0o1q"], [f"ST__MSc_NotMerged{suffix}"]),
#    ("st_bkg", ["st_bkg"], ["ST__MSc_Background"]),
#    ("mj", ["mj"], ["QCD"]),
#    ("vx", ["vx"], ["VJetsAndVV"]),
#    ("tt", ["tt_3q"], [f"TTbar__MSc_FullyMerged{suffix}"]),
#    ("st", ["sti"], [f"ST__MSc_FullyMerged{suffix}"]),
# ]

files = {}
hists = {}
if __name__ == "__main__":
    files["uhh2"] = uproot.open(paths["uhh2"])
    files["cf"] = [uproot.open(p) for p in paths["cf"]]

    if not os.path.exists("plots/run2__very_loose"):
        os.makedirs("plots/run2__very_loose")

    # -- fill individual hists

    hists = {}

    # channel
    for ch in channels:
        ch_uhh2 = uhh2_map["channel"][ch]
        hists_ch = hists[ch] = {}

        # pass/fail
        for pf in pass_fails:
            pf_uhh2 = uhh2_map["pass_fail"][pf]
            hists_ch_pf = hists_ch[pf] = {}

            cat_name = f"{ch}__17__pt_300_400__tau32_wp_very_loose_{pf}"

            # process
            for proc in processes:
                proc_uhh2 = uhh2_map["process"][proc]
                hists_ch_pf_proc = hists_ch_pf[proc] = {}

                # get UHH2 hist
                hists_ch_pf_proc["uhh2"] = files["uhh2"][f"CombBin-Main-{pf_uhh2}-{ch_uhh2}-UL17-pt_300to400/{proc_uhh2}"]  # noqa

                # get CF hist
                for f in files["cf"]:
                    try:
                        hists_ch_pf_proc["cf"] = f[f"bin_{cat_name}/{proc}"]
                    except uproot.KeyInFileError:
                        continue
                    else:
                        break
                if "cf" not in hists_ch_pf_proc:
                    raise ValueError("key not found")

                # convert to hist-type histograms
                hists_ch_pf_proc["uhh2"] = hists_ch_pf_proc["uhh2"].to_hist()
                hists_ch_pf_proc["cf"] = hists_ch_pf_proc["cf"].to_hist()

    # -- make compound hists
    for ch in channels:

        # all = pass + fail
        hists[ch]["all"] = {}
        for proc in processes:
            hists[ch]["all"][proc] = {}
            for src in ("cf", "uhh2"):
                hists[ch]["all"][proc][src] = hists[ch]["pass"][proc][src] + hists[ch]["fail"][proc][src]

        # subproc
        for pf in ("pass", "fail", "all"):
            for compound_proc, subprocs in compound_processes.items():
                hists[ch][pf][compound_proc] = {}
                for src in ("cf", "uhh2"):
                    hists[ch][pf][compound_proc][src] = sum(
                        hists[ch][pf][subproc][src]
                        for subproc in subprocs
                    )

    # -- make plots

    for ch in ("1m", "1e"):
        for pf in ("pass", "fail", "all"):
            if pf == "all":
                cat_name = f"{ch}__pt_300_400"
            else:
                cat_name = f"{ch}__pt_300_400__tau32_wp_very_loose_{pf}"

            # category and process instances (for label)
            cat_inst = config_2017.get_category(cat_name)

            for proc in processes + list(compound_processes):
                proc_inst = config_2017.get_process(proc)

                plot_comparison(
                    hists=[
                        hists[ch][pf][proc]["uhh2"],
                        hists[ch][pf][proc]["cf"],
                    ],
                    labels=["UHH2 (UL17)", "CF (UL17)"],
                    output_filename=f"plots/run2__very_loose/{ch}__{pf}__{proc}",
                    annotate_cfgs=[
                        dict(
                            text=cat_inst.label,
                            xy=(1, 0),
                            xycoords="axes fraction",
                            xytext=(-15, 15),
                            textcoords="offset points",
                            ha="right",
                            va="bottom",
                            fontsize=14,
                        ),
                        dict(
                            text=proc_inst.label,
                            xy=(1, 0),
                            xycoords="axes fraction",
                            xytext=(-15, 75),
                            textcoords="offset points",
                            ha="right",
                            va="bottom",
                            fontsize=16,
                        ),
                    ],
                    style_cfgs=[
                        {
                            "color": "#e2001a",
                            "linestyle": "dashed",
                            "linewidth": 2,
                        },
                        {
                            "color": "black",
                            "linestyle": "solid",
                            "linewidth": 3,
                        },
                    ],
                )

    # values = {k: hists[k].values() for k in hists}
