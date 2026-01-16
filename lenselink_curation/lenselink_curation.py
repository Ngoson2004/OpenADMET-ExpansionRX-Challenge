#!/usr/bin/env python3
import argparse, os, sys, json, glob
from pathlib import Path

def import_helpers(helper_path=None):
    mod_name = "data_curation_functions"
    if helper_path is None:
        helper_path = os.path.join(os.path.dirname(__file__), "data_curation_functions.py")
    if not os.path.isfile(helper_path):
        sys.exit(f"Could not find data_curation_functions.py at: {helper_path}")
    import importlib.util
    spec = importlib.util.spec_from_file_location(mod_name, helper_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def expand_inputs(globs):
    files = sorted({p for g in globs for p in glob.glob(g)})
    if not files:
        sys.exit("No input files matched your --inputs pattern(s).")
    return files

def with_suffix(path, extra):
    p = Path(path)
    # Preserve compression suffixes
    if p.suffix in (".gz", ".zip"):
        stem = p.name[: -(len(p.suffix))]
        return str(p.parent / f"{stem}.{extra}{p.suffix}")
    else:
        return str(p.parent / f"{p.stem}.{extra}.csv")

def parse_json_or_none(s):
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception as e:
        sys.exit(f"--transf-dict must be valid JSON (e.g. '{{\"fumic\":\"log10(x/(1-x))\"}}'). Error: {e}")

def main():
    ap = argparse.ArgumentParser(
        description="Run F2->F3→F4→F5 on multiple CSVs: standardise, transform, average, then append+pivot."
    )
    ap.add_argument("-i","--inputs", nargs="+", required=True, help="Input unpivoted CSVs (can be .csv, .csv.gz, .csv.zip).")
    ap.add_argument("-o","--output", required=True, help="Final pivoted CSV path (supports .csv, .csv.gz, .csv.zip).")
    ap.add_argument("--helpers", default=None, help="Path to data_curation_functions.py (default: alongside this script).")
    ap.add_argument("--workdir", default=None, help="Where to write intermediate F3/F4 files (default: <output_dir>/.tmp).")

    # F2 knobs
    ap.add_argument("--smiles-col", default="SMILES")
    ap.add_argument("--id-col", default="ID")
    ap.add_argument("--module-col", nargs="+", default="MODULE")
    ap.add_argument("--value-col", default="VALUE")
    ap.add_argument("--rm-col", nargs="+", default=[])
    ap.add_argument("--f2-out", default=None, help="Path to save intermediate file output by F2")

    # F3 knobs
    ap.add_argument("--skip-chem", action="store_true", help="Skip chemistry curation in F3.")
    ap.add_argument("--skip-data", action="store_true", help="Skip data curation in F3.")
    ap.add_argument("--value-prefix-col", default=None)
    ap.add_argument("--transf-dict", default=None,
                    help='JSON dict mapping module substrings→transform (default internal). Example: \'{"fumic":"log10(x/(1-x))"}\'')
    ap.add_argument("--transf-default", default="log10(x)",
                    choices=["x","log10(x)","log10(x/(1-x))","log10(x/(100-x))"])
    ap.add_argument("--no-transform-censored", action="store_true",
                    help="If set, censored values like '<0.01' are set to NA during transformation.")
    ap.add_argument("--f3-out", default=None, help="Path to save intermediate file output by F3")
    # ap.add_argument("--use-col", nargs="+", default=None, help="Value columns that will be brought to output. Only apply if skip data curation")

    # F4 knobs
    ap.add_argument("--aggregation", default="mean", choices=["mean","median"])
    ap.add_argument("--keep-only-numeric", action="store_true",
                    help="Keep only '=' qualified numeric transformed values for averaging.")
    ap.add_argument("--keep-U-and-C", action="store_true",
                    help="Keep uncensored number and qualifier columns in output (e.g., *_U, *_C).")
    ap.add_argument("--min-n", type=int, default=0, help="Drop MODULEs with fewer than N datapoints after averaging.")
    ap.add_argument("--remove-outliers", action="store_true",
                    help="When aggregation=mean, remove outliers by IQR before averaging.")
    ap.add_argument("--f4-out", default=None, help="Path to save intermediate file output by F4")

    # F5 knobs
    ap.add_argument("--central-tendency", default="TRANSFORMED_VALUE_MEAN",
                    choices=["TRANSFORMED_VALUE_MEAN","TRANSFORMED_VALUE_MEDIAN"])
    ap.add_argument("--no-transf-prefix", action="store_true", help="Do not prefix pivoted columns by TRANSFORMATION.")
    ap.add_argument("--tag-ids-by-input", action="store_true",
                    help="Prefix original_ID by per-file prefixes (required if same MODULE names appear across files).")
    ap.add_argument("--prefix-per-input", nargs="+", default=None,
                    help="List of per-file prefixes (length must equal number of inputs if provided).")

    args = ap.parse_args()
    mod = import_helpers(args.helpers)

    inputs = args.inputs
    out_dir = Path(args.output).resolve().parent
    workdir = Path(args.workdir or (out_dir / ".tmp"))
    workdir.mkdir(parents=True, exist_ok=True)

    # Prepare transformation dict for F3
    transf_dict = parse_json_or_none(args.transf_dict)  # may be None to use helper defaults

    f4_products = []
    for f in inputs:
        src = Path(f)
        f2_out = str(workdir / (src.name + ".f2.csv.gz")) if args.f3_out is None else args.f2_out
        f3_out = str(workdir / (src.name + ".f3.csv.gz")) if args.f3_out is None else args.f3_out
        f4_out = str(workdir / (src.name + ".f4.csv.gz")) if args.f4_out is None else args.f4_out

        mod.F2_csv_pivoted_to_unpivoted(
            input_pivoted_csv_file_path = str(f),
            output_unpivoted_csv_file_path = f2_out,
            SMILES_colname = args.smiles_col,
            original_ID_colname = args.id_col,
            columns_to_remove = args.rm_col,
            output_SMILES_colname = args.smiles_col,
            output_ID_colname = args.id_col,
            output_MODULE_colname = args.module_col,
            output_VALUE_colname = args.value_col,
        )

        # --- F3: standardise + transform + curate (writes unpivoted curated file)
        mod.F3_csv_unpivoted_to_standard_transformed_curated(
            input_unpivoted_csv_file_path=f2_out,
            output_unpivoted_csv_file_path=f3_out,
            original_ID_colname=args.id_col,
            do_chemistry_curation=not args.skip_chem,
            SMILES_colname=args.smiles_col,
            do_data_curation=not args.skip_data,
            MODULE_colname=args.module_col,
            MODULE_rename_dict=None,
            VALUE_colname=args.value_col,
            VALUE_prefix_colname=args.value_prefix_col,
            TRANSF_dict=transf_dict if transf_dict is not None else {'fumic':'log10(x/(1-x))','ppb':'log10(x/(100-x))','log':'x'},
            TRANSF_default=args.transf_default,
            transform_censored=not args.no_transform_censored,
        )

            # --- F4: average per-SMILES for each MODULE (writes averaged unpivoted file)
        mod.F4_csv_unpivoted_std_transf_cur_to_averaged(
            input_curated_unpivoted_csv_file_path=f3_out,
            output_averaged_unpivoted_csv_file_path=f4_out,
            keep_only_numeric=args.keep_only_numeric,
            keep_uncensored_and_qualifier_in_output=args.keep_U_and_C,
            min_number_data_points=args.min_n,
            aggregation_function=args.aggregation,
            remove_outliers=args.remove_outliers,
        )       
        f4_products.append(f4_out)


    # --- F5: append all averaged files & pivot to final wide dataset
    mod.F5_csv_unpivoted_std_avg_append_and_pivot(
        input_averaged_unpivoted_csv_file_path_list=f4_products,
        output_pivoted_csv_file_path=args.output,
        central_tendency_colname=args.central_tendency,
        use_transf_as_prefix=not args.no_transf_prefix,
        use_prefix_per_input_file_on_original_ID=args.tag_ids_by_input,
        prefix_per_input_file=args.prefix_per_input,
    )
        # mod.F5_AUXFUN_write_csv_from_dataframe_with_sparse_cols(
        #     dataframe = df,
        #     sparse_columns_names = list(df.columns),
        #     output_csv_file_full_path = args.output,
        #     overwrite=True
        # )

    print(f"\nAll done ✔  Final dataset: {args.output}")
    print(f"Intermediates in: {workdir}")

if __name__ == "__main__":
    main()
