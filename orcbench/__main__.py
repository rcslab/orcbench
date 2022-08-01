import click
import requests
import secrets

from .internals import run
from .internals.orcglobals import MODELS, MODEL_DIR, URL, CONTEXT_SETTINGS, RAW_DATA_DIR


def download_model(num):
    model_name = f"model-{num}.pickle"
    model_url = f"{URL}/{model_name}"
    response = requests.get(model_url)
    model_path = MODEL_DIR / model_name
    open(model_path, "wb").write(response.content)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--N", default=10, help="Sampling Group Size")
@click.option("--scale", default=0.25, help="Percentage of workload")
@click.option("--seed", default=secrets.randbits(32), help="Seed for sampling")
@click.option("--runtime", default=30, help="How long should trace run for")
@click.option("--out", default="trace.out", help="Output file for the trace")
def trace(n, scale, seed, runtime, out):
    job_map = run.get_jobs(n, scale, seed)
    trace = []
    total = 0
    for _, jobs in job_map.items():
        total += len(jobs)
    with click.progressbar(length=total, label="Running Trace") as bar:
        for _, jobs in job_map.items():
            for job in jobs:
                t = job.run_trace(runtime)
                trace.extend(t)
                bar.update(1)
    trace.sort(key=lambda x: x[0])
    click.echo(f"Total invocations: {len(trace)}")
    ids = set()
    with open(out, "w") as f:
        with click.progressbar(length=len(trace), label="Writing trace") as bar:
            for t in trace:
                ids.add(t[1])
                f.write(f"{t[0]}, {t[1]}, {t[2]}\n")
                bar.update(1)
    click.echo(f"Unique Functions: {len(ids)}")


@cli.command()
def download():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with click.progressbar(length=MODELS, label="Downloading Models") as bar:
        for i in range(0, MODELS):
            download_model(i)
            bar.update(1)


@cli.command()
@click.argument("model")
@click.option("--language", default="python")
def function(model, language):
    loaded_models = run.load_models()
    found = None
    for m in loaded_models:
        if m.name == model:
            found = model
    program_str = run.create_function(found, language=language)
    click.echo(program_str)


@cli.command()
@click.option("--data", default=RAW_DATA_DIR, type=click.Path(exists=True))
def create(data):
    from .models import create_models
    create_models(data)


if __name__ == "__main__":
    main_cli = click.CommandCollection(sources=[cli], context_settings=CONTEXT_SETTINGS)
    main_cli(prog_name="OrcBench")
