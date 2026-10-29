import os
import re
import yaml
import pandas as pd
import argparse
from medval.pipeline import MedVAL
from tqdm import tqdm
import dspy


def resolve_env(value):
    """Expand a `${VAR}` config value from the environment; return other values unchanged."""
    m = re.fullmatch(r"\$\{(\w+)\}", value) if isinstance(value, str) else None
    return os.environ.get(m.group(1)) if m else value


def judge_item(medval, x, debug, score_col):
    """Validate one example. Returns {output column: value}; lm_failure is empty on success."""
    try:
        result = medval.forward(reference=x["reference"], task=x["task"], candidate=x["candidate"])
        failure = "" if result["attack_prediction"] is not None else "no prediction returned"
    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")
        result = {"attack_prediction": None, "err": None, "reason": None}
        failure = f"{type(e).__name__}: {e}"
    if debug:
        print(dspy.inspect_history(n=2))
        exit()
    cols = {score_col: result["attack_prediction"], "lm_reasoning": result["reason"], "lm_failure": failure}
    if score_col == "lm_risk_grade":
        cols["lm_error_assessment"] = result["err"]
    return cols

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Name of the YAML file under the directory configs/")
    args = parser.parse_args()
    args.config = f"configs/{args.config}.yaml" if not args.config.endswith(".yaml") else args.config
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    api_key = resolve_env(config["api_key"])
    n_samples = int(config["n_samples"]) if config["n_samples"] is not None else None
    threshold = config.get("threshold", None)
    input_csv = config.get("input_csv", None)
    validator_type = config.get("validator_type", "medval")
    temperature = float(config.get("temperature", 0.0))
    max_tokens = int(config.get("max_tokens", 1000))
    output_dir = config.get("output_dir", "results")
    n_runs = int(config.get("n_runs", 1))
    retry_failed = bool(config.get("retry_failed", False))
    max_retries = int(config.get("max_retries", 2))
    # DSPy caches identical prompts, which would make repeated runs identical.
    cache = n_runs == 1

    if (config["data"] == "train") and (config["model"].startswith("local")):
        medval = MedVAL(config["tasks"], "local/" + config["local_model_path"], config["api_base"], api_key, config["data"], n_samples, config["debug"], config["method"], threshold, input_csv, validator_type, temperature, max_tokens, cache)
    else:
        medval = MedVAL(config["tasks"], config["model"], config["api_base"], api_key, config["data"], n_samples, config["debug"], config["method"], threshold, input_csv, validator_type, temperature, max_tokens, cache)

    if config["data"] == "train":
        medval.generator._compiled = True
        trainset, valset = medval.load_data()
        agents_path = f"agents/"
        output_dir = os.path.join(agents_path, f"{config['student_model'].split('/')[-1]}/")
        os.environ["DSPY_FINETUNEDIR"] = output_dir
        if (config["data"] == "train") and (config["model"].startswith("local")):
            medval_student = medval
        else:
            medval_student = MedVAL(config["tasks"], config["model"], config["api_base"], api_key, config["data"], n_samples, config["debug"], config["method"], threshold, input_csv)
            medval_student.student_model = "local/" + config["local_model_path"]
            medval_student._configure_lm()
            medval_student.generator._compiled = True
        
        if (config['student_model'].startswith("openai")) or (config['student_model'].startswith("azure")):
            optimizer = dspy.BootstrapFinetune(metric=medval.validator_metric, num_threads=config["num_threads"])
        else:
            optimizer = dspy.BootstrapFinetune(metric=medval.validator_metric, num_threads=config["num_threads"], train_kwargs=dict(num_train_epochs=config["num_epochs"], use_peft=True, use_quantization=True, output_dir=output_dir))
        optimized_validator = optimizer.compile(medval_student, teacher=medval, trainset=trainset)
        
    elif config["data"] == "test":
        medval.generator._compiled = True
        if config['model'].startswith("local"):
            agents_path = f"agents/"
            model_path = os.path.join(agents_path, f"{config['model'].split('/')[-1]}")
            medval.student_model = "local/" + config["local_model_path"]
            medval._configure_lm()
            medval.get_lm().launch()
        
        df, dataset = medval.load_data()
        score_col = "lm_safety_level" if validator_type == "safety" else "lm_risk_grade"
        out_path = medval.results_path(config["method"], output_dir)

        if retry_failed and os.path.exists(out_path):
            # Resume from saved results; only failed (#, run) rows are re-sent below.
            out = pd.read_csv(out_path)
            out["lm_failure"] = out["lm_failure"].fillna("") if "lm_failure" in out.columns else ""
            if "run" not in out.columns:
                out["run"] = 0
            print(f"Loaded {len(out)} existing rows from {out_path}")
        else:
            # One copy of the dataset per inference run, stored long with a `run` column.
            out = pd.concat([df.assign(run=run) for run in range(n_runs)], ignore_index=True)
            for col in [score_col, "lm_reasoning", "lm_failure", "lm_error_assessment"]:
                out[col] = None
            for k in tqdm(range(len(out))):
                for col, value in judge_item(medval, dataset[k % len(df)], config["debug"], score_col).items():
                    out.at[k, col] = value
            out = out.rename(columns={"reference": "input", "target": "reference_output", "candidate": "output"})

        if retry_failed:
            idx_by_id = {row_id: i for i, row_id in enumerate(df["#"])}
            for attempt in range(max_retries):
                failed = out.index[pd.to_numeric(out[score_col], errors="coerce").isna()]
                if len(failed) == 0:
                    break
                print(f"Retry pass {attempt + 1}/{max_retries}: {len(failed)} failed (#, run) rows")
                for k in tqdm(failed):
                    x = dataset[idx_by_id[out.at[k, "#"]]]
                    for col, value in judge_item(medval, x, config["debug"], score_col).items():
                        out.at[k, col] = value

        if validator_type == "safety":
            out = out[["#", "id", "task", "run", "input", "reference_output", "output", "lm_reasoning", "physician_error_assessment", "lm_safety_level", "physician_risk_grade", "lm_failure"]]
        else:
            out = out[["#", "id", "task", "run", "input", "reference_output", "output", "lm_reasoning", "lm_error_assessment", "physician_error_assessment", "lm_risk_grade", "physician_risk_grade", "lm_failure"]]
        medval.save_results(out, config["method"], output_dir)

        if config['model'].startswith("local"):
            medval.get_lm().kill()