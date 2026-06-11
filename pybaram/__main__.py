# -*- coding: utf-8 -*-
from argparse import ArgumentParser, FileType


def process_import(args):
    from pybaram.api.io import import_mesh

    import_mesh(args.inmesh, args.outmesh, args.scale)


def process_part(args):
    from pybaram.api.io import partition_mesh

    partition_mesh(args.mesh, args.out, args.npart, args.soln)


def process_export(args):
    from pybaram.api.io import export_soln

    export_soln(
        args.mesh, args.soln, args.out, 
        args.surface, args.list_surfaces
        )


def process_run(args):
    from pybaram.api.simulation import run
    from pybaram.inifile import INIFile
    from pybaram.readers.native import NativeReader

    mesh = NativeReader(args.mesh)
    cfg = INIFile(args.ini)

    run(mesh, cfg)


def process_restart(args):
    from pybaram.api.simulation import restart
    from pybaram.inifile import INIFile
    from pybaram.readers.native import NativeReader

    mesh = NativeReader(args.mesh)
    soln = NativeReader(args.soln)
    
    # Config file
    if args.ini:
        cfg = INIFile(args.ini)
    else:
        cfg = INIFile()
        cfg.fromstr(soln['config'])

    restart(mesh, soln, cfg)


def main():
    ap = ArgumentParser(prog='pybaram')
    sp = ap.add_subparsers(dest='cmd', help='sub-command help')

    # Common options
    ap.add_argument('--verbose', '-v', action='count')

    # Import command
    ap_import = sp.add_parser('import', help='import --help')
    ap_import.add_argument('inmesh', help='input mesh file')
    ap_import.add_argument('outmesh', help='output mesh file')
    ap_import.add_argument('-s', '--scale', type=float, default=1,
                           help='scale mesh')
    ap_import.set_defaults(process=process_import)

    # Partition command
    ap_part = sp.add_parser('partition', help='partition --help')
    ap_part.add_argument('npart', help='number of partition')
    ap_part.add_argument('mesh', help='mesh file')
    ap_part.add_argument('soln', nargs='*', type=str, help='solution file')
    ap_part.add_argument('out', help='partitioned mesh file')
    ap_part.set_defaults(process=process_part)

    # Run command
    ap_run = sp.add_parser('run', help='run --help')
    ap_run.add_argument('mesh', type=str, help='mesh file')
    ap_run.add_argument('ini', type=str, help='config file')
    ap_run.set_defaults(process=process_run)

    # Run restart
    ap_restart = sp.add_parser('restart', help='run --help')
    ap_restart.add_argument('mesh', type=str, help='mesh file')
    ap_restart.add_argument('soln', type=str, help='solution file')
    ap_restart.add_argument('ini', nargs='?', type=str, help='config file')
    ap_restart.set_defaults(process=process_restart)

    # Export command
    ap_export = sp.add_parser('export', help='export --help')
    ap_export.add_argument('mesh', help='mesh file')
    ap_export.add_argument('soln', nargs='?', help='solution file')
    ap_export.add_argument('out', nargs='?', help='output file')
    ap_export.set_defaults(process=process_export)

    # surface option
    ap_export.add_argument(
        "-s", "--surface",
        type=str,
        help='Export surface data by boundary name; use commas to specify multiple boundaries (e.g., wall or wall,inlet)'
    )

    ap_export.add_argument(
        "--list-surfaces",
        action="store_true",
        help="List available surface names in mesh"
    )

    # Parse the arguments
    args = ap.parse_args()

    # Invoke the process method
    if hasattr(args, 'process'):
        args.process(args)
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
