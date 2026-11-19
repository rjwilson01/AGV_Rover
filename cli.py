"""Command-line definition (kept free of heavy imports so the README generator can read it)."""
import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Snake AGV simulation")
    parser.add_argument("preset", nargs="?", default=None, help="controller preset, e.g. VariableDrawbarController")
    parser.add_argument(
        "--tests", default=None, metavar="SPEC",
        help="test sequence, e.g. 200-a,b,c,100-d,e,100+f,g  (seconds, then - = reset to defaults / + = keep "
             "previous files, then files in configs/test/; each group runs once its files are loaded)",
    )
    parser.add_argument(
        "--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
        help="override one config value (TOML syntax); beats every test file. Repeatable.",
    )
    parser.add_argument("--plotjuggler", action="store_true", help="launch PlotJuggler (off by default)")
    parser.add_argument("--selfcheck", action="store_true", help="measure the sign conventions with short open-loop moves, then exit")
    parser.add_argument("--view", action="store_true",
                        help="MANUAL review: show the model (preset + the first --tests group) with physics running and every actuator "
                             "on keyboard hot keys (listed on screen); no controller, CSV or PlotJuggler")
    parser.add_argument("--dump-xml", default=None, metavar="FILE", help="with --view: also write the generated MJCF to FILE")
    parser.add_argument("--drive", default=None, metavar="SPEC",
                        help="open-loop lead profile for impulse/step-response tests: start:end:speed:curvature,... "
                             "(replaces path following; followers run their normal algorithm)")
    parser.add_argument("--no-viewer", action="store_true", help="run headless, faster than real time")
    return parser
