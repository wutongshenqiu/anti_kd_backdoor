import json
from argparse import ArgumentParser
from pathlib import Path

from pyecharts import options as opts
from pyecharts.charts import Line


def collect_sub_dirs(base_dir: Path) -> list[Path]:
    return list(filter(lambda x: x.is_dir(), base_dir.iterdir()))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--work_dir',
                        '-w',
                        type=Path,
                        help='Path of work directory')
    parser.add_argument(
        '--base_dir',
        '-b',
        type=Path,
        help='Base directory, each sub directory must be a work directory')

    args = parser.parse_args()
    print(args)
    if not ((args.work_dir is None) ^ (args.base_dir is None)):
        raise ValueError(
            'One and only one of `work_dir` and `base_dir` should be specified'
        )

    if args.work_dir is not None:
        assert args.work_dir.exists()
        work_dir_list = [args.work_dir]
    else:
        assert args.base_dir.exists()
        work_dir_list = collect_sub_dirs(args.base_dir)

    for work_dir in work_dir_list:
        print(f'Generating graph for directory: {work_dir}')
        result_file = work_dir / 'results.json'
        if not result_file.exists():
            print('Result file does not exist, process will be ignored')
            continue

        with result_file.open('r', encoding='utf8') as f:
            result = json.loads(f.read())

        line = (Line(opts.InitOpts(
            width='1600px', height='900px')).set_global_opts(
                title_opts=opts.TitleOpts(
                    title='ASR with different transparency'),
                tooltip_opts=opts.TooltipOpts(trigger='axis'),
                toolbox_opts=opts.ToolboxOpts(is_show=True),
                xaxis_opts=opts.AxisOpts(name='transparency'),
                yaxis_opts=opts.AxisOpts(name='ASR')))

        # add xaxis
        for model_name, v in result.items():
            asr_list = v['asr']
            transparency_list = [str(asr['transparency']) for asr in asr_list]
            break
        line.add_xaxis(xaxis_data=transparency_list)

        # add yaxis
        for model_name, v in result.items():
            asr_list = v['asr']
            top1_list = [asr['top1'] for asr in asr_list]

            line.add_yaxis(
                # series_name=model_name,
                series_name='',
                y_axis=top1_list,
                label_opts=opts.LabelOpts(is_show=False),
            )

        line.render(work_dir / 'charts.html')
