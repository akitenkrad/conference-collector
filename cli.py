import click
from os import PathLike
from pathlib import Path
import json
from tqdm import tqdm

from neurips.neurips_2021 import NeurIPS_2021

@click.group()
def cli():
    pass

@cli.command()
@click.option('--out-path', type=click.Path(), help='path to output json file')
def neurips_2021(out_path:PathLike):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    neurips = NeurIPS_2021()
    papers = neurips.collect()
    papers_all = []
    for paper in tqdm(papers):
        papers_all.append(neurips.get_detail(paper))

    json.dump(papers_all, open(out_path, 'wt', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('Done.')

if __name__ == '__main__':
    cli()
