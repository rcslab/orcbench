import click
import secrets

from .internals import run
from .internals.orcglobals import CONTEXT_SETTINGS, RAW_DATA_DIR

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
@click.argument("model")
@click.option("--language", default="python")
def function(model, language):
    loaded_models = run.load_models()
    found = None
    for m in loaded_models:
        if m.name == model:
            found = m
    if found is None:
        click.echo("No model found of that name, available models are:")
        click.echo(",".join([ m.name for m in loaded_models ]))
        exit(0)

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
