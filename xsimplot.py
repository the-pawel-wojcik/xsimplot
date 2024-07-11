#!/usr/bin/env python

import argparse
import os
import sys
import math as m
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as ticker
import numpy as np
import tomllib


DISREGARD_INTENSITY = 1e-20

COLORS = [color for color in mcolors.TABLEAU_COLORS.keys()]
FONTSIZE = 12
CM2INCH = 1/2.54

ZEKE_LINES_CM = [0.0, 616.8, 1219.9, 1815.3, 2465.5]
ZEKE_ADIABATIC_IE_CM = 101020.5
ZEKE_2ND_ADIABATIC_IE_CM = 102110.1
ZEKE_DISSOCIATION_THRESHOLD = ZEKE_ADIABATIC_IE_CM + 4898  # (3)
CM2eV = 1./8065.543937
eV2CM = 8065.543937

PYRAZINE_ABSORPTION_ORIGIN_CM = 30876


def parse_command_line():
    # Parse the command line arguments.

    parser = argparse.ArgumentParser(
        description="Plot output of the xsim program.\nUse the -n flag and"
        " work with the fort.20 output files if you do not have xsim parser"
        " installed.")

    parser.add_argument("output_files",
                        help="List of xsim output files "
                        "(requires xsim parser). "
                        "If the -n flag is up, it's a list of fort.20 files "
                        "(no extra programs required).",
                        nargs="+")

    parser.add_argument("-a", "--annotate",
                        help="Place string with the annotation on the figure."
                        "Use 'a' as the first letter of the string to append"
                        " the annotation generated by other flags; use 'o' to"
                        " overwrite.",
                        default="",
                        type=str)

    parser.add_argument("-c", "--config",
                        help="Pick config file.",
                        default="xsimplot.toml")

    parser.add_argument("-e", "--envelope",
                        help="Add a Lorenzian envelope to every peak.",
                        type=str,
                        default=None,
                        choices=['stack', 'overlay', 'sum only'])

    parser.add_argument("-g", "--gamma",
                        help="Gamma in the Lorenzian:\n" +
                        "(0.5 * gamma)**2 / ((x - x0)**2 + (0.5 * gamma) **2)",
                        type=float,
                        default=None)

    parser.add_argument("-k", "--horizonal_minor_ticks_2nd_axis",
                        help="Specify the interval at which the minor ticks"
                        " should appear.",
                        type=float,
                        default=None)

    parser.add_argument("-K", "--horizonal_minor_ticks",
                        help="Specify the interval at which the minor ticks"
                        " should appear.",
                        type=float,
                        default=None)

    help = "Match some of the plot properties (like xlims or output location)"
    help += " to the selected molecule."
    parser.add_argument("-m", "--molecule",
                        help=help,
                        type=str,
                        choices=["ozone", "ozone_zeke", "ozone_dyke",
                                 "ozone_no_cpl", "pyrazine", "caoph"])

    parser.add_argument("-n", "--no_parser",
                        help="Use the fort.20 outputs of xsim as the source of"
                        " spectrum information.",
                        default=False,
                        action='store_true')

    parser.add_argument("-r", "--scale_factor",
                        help="Scale the figure size with the factor.",
                        default=None,
                        type=float)

    parser.add_argument("-t", "--sticks_off",
                        help="Do NOT show stick spectrum.",
                        default=False,
                        action="store_true")

    parser.add_argument("-v", "--verbose",
                        help="Annotate with # of Lanczos it and basis size.",
                        default=False,
                        action="store_true")

    save = parser.add_mutually_exclusive_group()

    save.add_argument("-d", "--dont_save",
                      help="Just show the figure. Don't save it yet.",
                      default=False,
                      action="store_true")

    save.add_argument("-f", "--filename",
                      help="Save figure as filename.",
                      default=None)

    shift = parser.add_mutually_exclusive_group()

    help = "Shift simulated spectrum to align the first peak at <match_origin> eV."
    shift.add_argument("-o", "--match_origin",
                       default=None,
                       help=help)

    shift.add_argument("-s", "--shift_eV",
                       help="Positive shift moves peak towards lower energy.",
                       type=float)

    parser.add_argument("-x", "--second_axis",
                        help="Add a second energy axis; pick units. If you"
                        " want to see the offset from the first peak add"
                        " 'offset'.",
                        type=str,
                        choices=["cm", "cm offset", "nm"],
                        default=None)

    parser.add_argument("-y", "--show_yaxis_ticks",
                        help="Show ticks on the yaxis.",
                        default=False,
                        action='store_true')

    args = parser.parse_args()
    return args


def height_one_lorenzian(x, x0, gamma):
    return lorenzian(x, x0, gamma) * m.pi * 0.5 * gamma


def lorenzian(x, x0, gamma):
    '''
    This definition of a Lorenzian comes from Eq. 2.40

    Köppel, H., W. Domcke, and L. S. Cederbaum. “Multimode Molecular Dynamics
    Beyond the Born-Oppenheimer Approximation.” In Advances in Chemical Physics
    edited by I. Prigogine and Stuart A. Rice, LVII:59–246, 1984.
    https://doi.org/10.1002/9780470142813.ch2.

    gamma corresponds to full width at half max (FWHM)
    '''
    return 1.0 / m.pi * (0.5 * gamma / ((x - x0)**2 + (0.5 * gamma) ** 2))


def turn_spectrum_to_xs_and_ys(spectrum):
    xs = []
    ys = []
    for spectral_point in spectrum:
        xs += [spectral_point['Energy (eV)']]
        ys += [spectral_point['Relative intensity']]

    return xs, ys


def lorenz_intensity(x, gamma, xsim_output):
    out = 0.0
    for spectral_point in xsim_output:
        energy_eV = spectral_point['Energy (eV)']
        intensity = spectral_point['Relative intensity']
        out += intensity * height_one_lorenzian(x, energy_eV, gamma)
    return out


def stem_xsim_output(xsim_output, ax, color: str = 'k'):
    for spectral_point in xsim_output:
        energy_eV = spectral_point['Energy (eV)']
        intensity = spectral_point['Relative intensity']
        ax.vlines(x=energy_eV, ymin=0.0, ymax=intensity, colors=color)


def stem_ozone_zeke(xsim_output, ax, color: str = 'k'):
    for spectral_point in xsim_output:
        energy_eV = spectral_point['Energy (eV)']
        intensity = 1.0
        ax.vlines(x=energy_eV, ymin=0.0, ymax=intensity, colors=color)


def get_xsim_outputs_from_fort20(args):
    xsim_outputs = []
    lanczos = None
    for file_idx, fort20_file in enumerate(args.output_files):

        with open(fort20_file, 'r') as f:
            fort20 = f.readlines()

        loc_lanczos = int(fort20[0].split()[0])
        if lanczos is None:
            lanczos = loc_lanczos
        elif loc_lanczos != lanczos:
            print("Warning: the number of Lanczos iterations in the input"
                  f" file #{file_idx} differs from the previous files.",
                  file=sys.stderr)

        xsim_output = []
        first_peak_cm = float(fort20[1].split()[0]) * eV2CM
        for peak in fort20[1:]:
            peak = peak.split()
            try:
                intensity = float(peak[1])
            except ValueError:
                # for tiny tiny intensities xsim produces garbage, i.e.:
                # 6.86644579832093   0.159775868164848-105  39294.6062024455
                exponent = int(peak[1][-4:])
                if exponent < -99:
                    continue
                print("Error: cannot process the line:" + peak,
                      file=sys.stderr)
                sys.exit(1)
            if intensity < DISREGARD_INTENSITY:
                continue
            offset = float(peak[2])
            data = {
                'Energy (eV)': float(peak[0]),
                'Energy (cm-1)': first_peak_cm + offset,
                'Offset (cm-1)': offset,
                'Relative intensity': float(peak[1]),
            }
            xsim_output += [data]

        xsim_outputs += [xsim_output]

    return xsim_outputs, lanczos


def get_xsim_outputs(args):
    from parsers.xsim import parse_xsim_output
    xsim_outputs = []
    basis = None
    lanczos = None
    for file_idx, out_file in enumerate(args.output_files):
        with open(out_file) as f:
            xsim_data = parse_xsim_output(f)

        xsim_output = xsim_data['spectrum_data']
        xsim_outputs += [xsim_output]

        loc_basis = xsim_data['basis']
        if basis is None:
            basis = loc_basis
        elif basis != loc_basis:
            print("Warning: Outputs use different basis sets.",
                  file=sys.stderr)

        loc_lanczos = xsim_data['Lanczos']
        if lanczos is None:
            lanczos = loc_lanczos
        elif lanczos != loc_lanczos:
            print("Warning! Outputs use different # of Lanczos iterations.",
                  file=sys.stderr)

    return xsim_outputs, basis, lanczos


def find_shift(xsim_outputs, args, config):
    """
    Find shift (in eV) which will be applied to the spectrum.
    The shift is
    """

    # Command line args are the most important
    if args.shift_eV is not None:
        return args.shift_eV

    first_peak_position = None

    if 'match_origin' in config:
        first_peak_position = config['match_origin']

    if args.match_origin is not None:
        first_peak_position = args.match_origin

    # if args.molecule is not None:
    #     if args.molecule == "ozone":
    #         # Position of the first ionization energy of ozone
    #         first_peak_position = ZEKE_ADIABATIC_IE_CM * CM2eV
    #     elif args.molecule == "ozone_zeke":
    #         # Position of the first ionization energy of ozone
    #         first_peak_position = ZEKE_ADIABATIC_IE_CM * CM2eV
    #     elif args.molecule == "ozone_dyke":
    #         # Position of the first ionization energy of ozone
    #         first_peak_position = ZEKE_ADIABATIC_IE_CM * CM2eV
    #     # elif args.molecule == "ozone_no_cpl":
    #     #     # Position of the first ionization energy of ozone
    #     #     first_peak_position = ZEKE_ADIABATIC_IE_CM * CM2eV
    #     elif args.molecule == "pyrazine":
    #         first_peak_position = PYRAZINE_ABSORPTION_ORIGIN_CM * CM2eV

    if first_peak_position is not None:
        origins = []
        for xsim_output in xsim_outputs:
            min_element = min(xsim_output, key=lambda x: x['Energy (eV)'])
            origins.append(min_element['Energy (eV)'])
        origin_eV = min(origins)
        shift_eV = origin_eV - first_peak_position
        return shift_eV

    return None


def find_left_right_gamma(xsim_outputs, args, config, how_far: int = 4):
    """
    Find the positions of the lowest energy and highest energy peaks and return
    values how_far*gamma away from them as the plot limits.
    """

    gamma_eV = find_gamma(args, config)

    # Commnad line is the most important
    if args.molecule is not None and args.molecule not in [
            "ozone_dyke", "ozone", "ozone_zeke", "ozone_no_cpl"
    ]:
        # if args.molecule == "ozone" or args.molecule == "ozone_zeke":
        #     # First photoelectron band of ozone
        #     left = 12.225
        #     right = 13.375

        # if args.molecule == "ozone_no_cpl":
        #     # First photoelectron band of ozone
        #     left = 12.225
        #     right = 13.375

        if args.molecule == "pyrazine":
            # First absorption band of pyrazine
            # left = 3.75
            # right = 4.25
            left = 3.8
            right = 4.0
            # # For 1st band absorption
            # left = 3.5
            # right = 4.5

        elif args.molecule == "caoph":
            left = 1.95
            right = 2.35

        return (left, right, gamma_eV)

    # Config file is the second most important resource
    if 'xlims' in config:
        left = config['xlims']['left']
        right = config['xlims']['right']
        return (left, right, gamma_eV)

    # Finally find some default values
    mins = list()
    maxes = list()

    for xsim_output in xsim_outputs:
        min_element = min(xsim_output, key=lambda x: x['Energy (eV)'])
        max_element = max(xsim_output, key=lambda x: x['Energy (eV)'])
        mins.append(min_element['Energy (eV)'])
        maxes.append(max_element['Energy (eV)'])

    minimum = min(mins)
    maximum = max(maxes)
    left = minimum - how_far * gamma_eV
    right = maximum + how_far * gamma_eV

    return (left, right, gamma_eV)


def apply_shift(xsim_outputs, shift_eV):
    if shift_eV is None:
        return xsim_outputs

    for output in xsim_outputs:
        for peak in output:
            peak['Energy (eV)'] -= shift_eV

    return xsim_outputs


def add_envelope(ax, args, config, xsim_outputs, xlims, gamma):

    envelope_type = None
    if 'envelope' in config:
        envelope_type = config['envelope']
    # Command line args override config options
    if args.envelope is not None:
        envelope_type = args.envelope

    if envelope_type is None:
        return 0.0

    npoints = 500
    xs = np.linspace(xlims[0], xlims[1], npoints)
    accumutated_ys = np.zeros_like(xs)

    for file_idx, xsim_output in enumerate(xsim_outputs):
        if file_idx == len(COLORS) or file_idx > len(COLORS):
            print("Too many colors already.", file=sys.stderr)
            sys.exit(1)

        state_spectrum = [lorenz_intensity(x, gamma, xsim_output) for x in xs]
        state_spectrum = np.array(state_spectrum)
        color = COLORS[file_idx]
        if envelope_type == "stack":
            ax.fill_between(xs, accumutated_ys + state_spectrum,
                            accumutated_ys, color=color, alpha=0.2)
        elif envelope_type == "overlay":
            ax.fill_between(xs, state_spectrum, np.zeros_like(xs), color=color,
                            alpha=0.2)
        accumutated_ys += state_spectrum

    dashed = (0, (5, 5))
    densly_dashdotted = (0, (3, 1, 1, 1))
    # Plot the total spectrum extra for overlay
    if envelope_type in ["overlay", "sum only"]:
        if args.molecule == "ozone_dyke":
            # Dyke-like
            # ax.fill_between(xs, accumutated_ys, np.zeros_like(xs),
            #                 color='tab:green', alpha=0.2, label='simulation')
            ax.plot(xs, accumutated_ys, color='tab:blue', lw=1.5,
                    ls=densly_dashdotted, label='simulation')
            ax.plot([], [], color='k', lw=1.5, label="experiment")
        else:
            ax.plot(xs, accumutated_ys, color='tab:gray', lw=1)

    fig_max_y = np.max(accumutated_ys)

    if args.molecule == "ozone_dyke":
        ax.legend()

    return fig_max_y


def add_peaks(ax, args, config, xsim_outputs, xlims):
    sticks_off = False
    if 'sticks_off' in config:
        sticks_off = config['sticks_off']
    if args.sticks_off is True:
        sticks_off = True

    if sticks_off is True:
        return 0.0

    peaks_maxima = []
    for file_idx, xsim_output in enumerate(xsim_outputs):
        if file_idx == len(COLORS) or file_idx > len(COLORS):
            print("Too many colors already.", file=sys.stder)
            sys.exit(1)

        my_peaks = [
            peak for peak in xsim_output if
            peak['Energy (eV)'] >= xlims[0] and peak['Energy (eV)'] <= xlims[1]
        ]
        if args.molecule == "ozone_zeke":
            stem_ozone_zeke(my_peaks, ax, COLORS[file_idx])
        else:
            stem_xsim_output(my_peaks, ax, COLORS[file_idx])

        max_peak = max(my_peaks, key=lambda x: x['Relative intensity'])
        peaks_maxima.append(max_peak['Relative intensity'])

    return max(peaks_maxima)


def add_info_text(ax, args, config, shift_eV, basis, lanczos, gamma):
    info_kwargs = {'horizontalalignment': 'left',
                   'verticalalignment': 'top',
                   'fontsize': FONTSIZE,
                   'color': 'k',
                   'transform': ax.transAxes,
                   }

    text = ""

    envelope_type = None
    if 'envelope' in config:
        envelope_type = config['envelope']
    if args.envelope is not None:
        envelope_type = args.envelope

    if envelope_type is not None:
        text = r'$\gamma = ' + f'{gamma:.3f}$\n'

    if shift_eV is not None:
        text += f'$s = {shift_eV:.2f}$'

    verbose = False
    if 'verbose' in config:
        verbose = config['verbose']
    if args.verbose is True:
        verbose = True

    if verbose is True:
        if basis is not None:
            text += f'\nBasis: {basis.split()[0]}'
        text += f'\nLanczos: {lanczos}'

    annotation = ""
    if 'annotate' in config:
        annotation = config['annotate']
    if args.annotate != "":
        annotation = args.annotate

    if annotation != "":
        # append
        if annotation[0] == "a":
            text += "\n" + annotation[1:]
        # overwrite
        elif annotation[0] == "o":
            text = annotation[1:]
        else:
            print("Warning: The annotation texts needs to start with either"
                  " 'a' or 'o', see help for details.", file=sys.stderr)

    ax.text(0.01, 0.99, text, **info_kwargs)


def add_caoph_lines(ax, top_feature):
    CAOPH_LINES = [
        {'pos_cm': 0.0, 'height': 660, 'lbl': 'A'},
        {'pos_cm': 0.0, 'height': 660, 'lbl': 'A'},
    ]

    max_height = max([x['height'] for x in CAOPH_LINES])

    # top_feature = 0.038
    # For now I print only the first band
    top_feature = 0.004
    # For the 2200 cm offset
    # top_feature = 0.022

    step = 0.075 * top_feature
    y_top = 0.85 * top_feature
    y_bottom = 0.825 * top_feature

    line_kwargs = {'color': 'gray',
                   'linestyles': 'solid',
                   'linewidths': 1,
                   'alpha': 0.2}
    text_kwargs = {'va': 'center',
                   'ha': 'center'}

    for line in CAOPH_LINES:
        pos_cm = line['pos_cm']
        lttr = line['lbl']
        if 'height' in line:
            height = line['height'] / max_height * top_feature
            y_text = height
            y_peak = height
        else:
            series = line['series']
            y_text = y_top - series * step
            y_peak = y_bottom - series * step
        x_cm = pos_cm + PYRAZINE_ABSORPTION_ORIGIN_CM
        x_eV = x_cm * CM2eV
        ax.text(x_eV, y_text, lttr, **text_kwargs)
        ax.vlines([x_eV], 0.0, [y_peak], **line_kwargs)


def add_pyrazine_lines(ax, top_feature):
    PYRAZINE_LINES = [
        {'pos_cm': 0.0, 'height': 660, 'lbl': '0', 'series': 0},
        {'pos_cm': 383, 'height': 487, 'lbl': r'$10a ^1$', 'series': 1},
        # {'pos_cm': 467, 'lbl': r'$16 b^2$', 'series': 3},
        {'pos_cm': 517, 'height': 122, 'lbl': r'$5 ^1$', 'series': 4},
        {'pos_cm': 583, 'height': 350, 'lbl': r'$6a ^1$', 'series': 2},
        {'pos_cm': 823, 'height': 273, 'lbl': r'$10 a^2$', 'series': 1},
        {'pos_cm': 945, 'height': 231, 'lbl': r'$10 a^1 6 a^1$', 'series': 3},
        # {'pos_cm': 1167, 'lbl': r'$6a ^2$', 'series': 2},
    ]

    max_height = max([x['height'] for x in PYRAZINE_LINES if 'height' in x])

    # top_feature = 0.038
    # For now I print only the first band
    top_feature = 0.004
    # For the 2200 cm offset
    # top_feature = 0.022

    step = 0.075 * top_feature
    y_top = 0.85 * top_feature
    y_bottom = 0.825 * top_feature

    line_kwargs = {'color': 'gray',
                   'linestyles': 'solid',
                   'linewidths': 1,
                   'alpha': 0.2}
    text_kwargs = {'va': 'center',
                   'ha': 'center'}

    for line in PYRAZINE_LINES:
        pos_cm = line['pos_cm']
        lttr = line['lbl']
        if 'height' in line:
            height = line['height'] / max_height * top_feature
            y_text = height
            y_peak = height
        else:
            series = line['series']
            y_text = y_top - series * step
            y_peak = y_bottom - series * step
        x_cm = pos_cm + PYRAZINE_ABSORPTION_ORIGIN_CM
        x_eV = x_cm * CM2eV
        ax.text(x_eV, y_text, lttr, **text_kwargs)
        ax.vlines([x_eV], 0.0, [y_peak], **line_kwargs)


def add_ZEKE_lines(ax, top_feature):
    y_top = 1.1 * top_feature
    y_bottom = 1.075 * top_feature
    xs = []
    for lttr, pos in zip(['B', 'C', 'D', 'E', 'F'], ZEKE_LINES_CM):
        x = pos + ZEKE_ADIABATIC_IE_CM
        x_ev = x * CM2eV
        xs.append(x_ev)
        ax.text(x_ev, y_top, lttr, va='center', ha='center')

    ax.vlines(xs, 0.0, y_bottom, color='gray', linestyles='solid',
              linewidths=1, alpha=0.2)

    # ax.vlines(xs, y_bottom, y_top, color = 'k')
    # ax.hlines(y_top, xs[0], xs[-1], color = 'k')
    # add the 2nd IE line

    # x = ZEKE_2ND_ADIABATIC_IE_CM
    # x_ev = x * CM2eV
    # lttr = r'$\tilde{A} ^+ (0,0,0)$'
    # lttr = r'$\tilde{A} ^+$'
    # # lttr = r'$IE _2$'
    # # y_bottom *= 0.9
    # # y_top *= 0.9
    # ax.text(x_ev, y_top, lttr, va='center', ha='center')
    # ax.vlines(x_ev, 0.0, y_bottom, color='gray', linestyles='solid',
    #           linewidths=1, alpha=0.2)

    x = ZEKE_DISSOCIATION_THRESHOLD
    x_ev = x * CM2eV
    lttr = r'$D _0$'
    ax.text(x_ev, y_top, lttr, va='center', ha='center')
    ax.vlines(x_ev, 0.0, y_bottom, color='gray', linestyles='solid',
              linewidths=1, alpha=0.2)

    CONICAL_INTERSECTION = 104194.3
    x = CONICAL_INTERSECTION
    x_ev = x * CM2eV
    lttr = 'CI'
    ax.text(x_ev, y_top, lttr, va='center', ha='center')
    ax.vlines(x_ev, 0.0, y_bottom, color='gray', linestyles=':',
              linewidths=1, alpha=0.2)


def add_assignmnet_to_ZEKE_lines(ax, top_feature):

    uncoupled_like = [
        [0, "A000"],
        [618, "A010"],
        # [915, r"A001$^*$"],
        [915, r"A001"],
        [1076, "A100"],
        [1222, "A020"],
        [1368, "B000"],
        [1684, "A110"],
    ]

    leading_order = [
        [1796, "A030", 0.87],
        [1921, "B010", 0.74],
        [2275, "A120", 0.74],
        [2362, "A040", 0.38],
        [2886, "A130", 0.66],
        [3045, "A050", 0.58],
    ]

    text_kw = dict(va='bottom', ha='left', rotation=35,
                   rotation_mode='anchor')

    y_top = top_feature
    y_top = 1.1 * top_feature
    y_bottom = 1.075 * top_feature
    xs = []
    for peak in uncoupled_like:
        pos = peak[0]
        name = peak[1]
        assignmnet = ""
        if name[0] == "A":
            assignmnet += r"A$_1("
        elif name[0] == "B":
            assignmnet += r"B$_2("
        assignmnet += name[1:4]
        assignmnet += ")$"
        if len(name) > 4:
            assignmnet += name[4:]

        x = pos + ZEKE_ADIABATIC_IE_CM
        x_ev = x * CM2eV
        xs.append(x_ev)
        ax.text(x_ev, y_top, assignmnet, **text_kw)

    # ax.vlines(xs, 0.0, y_bottom, color='gray', linestyles='solid',
    #           linewidths=1, alpha=0.2)

    for peak in leading_order[:-2]:
        pos = peak[0]
        name = peak[1]
        percentage = peak[2]
        assignmnet = f"{100*percentage**2:.0f}%"
        if name[0] == "A":
            assignmnet += r"A$_1$("
        elif name[0] == "B":
            assignmnet += r"B$_2$("
        assignmnet += name[1:5]
        assignmnet += ")"

        x = pos + ZEKE_ADIABATIC_IE_CM
        x_ev = x * CM2eV
        xs.append(x_ev)
        ax.text(x_ev, y_top, assignmnet, **text_kw)

    text_kw = dict(va='bottom', ha='right', rotation=-20,
                   rotation_mode='anchor')

    for peak in leading_order[-2:]:
        pos = peak[0]
        name = peak[1]
        percentage = peak[2]
        assignmnet = f"{100*percentage**2:.0f}%"
        if name[0] == "A":
            assignmnet += r"A$_1$("
        elif name[0] == "B":
            assignmnet += r"B$_2$("
        assignmnet += name[1:5]
        assignmnet += ")"

        x = pos + ZEKE_ADIABATIC_IE_CM
        x_ev = x * CM2eV
        xs.append(x_ev)
        ax.text(x_ev, y_top, assignmnet, **text_kw)


def add_no_cpl_lines(ax):

    lines = {
        "A000": 0, "A010": 628, "A100": 1083, "A020": 1249, "B000": 1412,
        "A110": 1708, "A030": 1863, "B010": 2061, "A120": 2325,
        "A040": 2470, "B020": 2708, "A130": 2936, "A050": 3068,
        "B110": 3247, "B030": 3353, "A140": 3539, "A060": 3656,
        "B120": 3888, "B040": 3996, "A150": 4134, "A070": 4234,
        "B130": 4527, "B050": 4636, "A080": 4801, "B140": 5163,
        "B060": 5275, "A090": 5353, "B150": 5802, "B070": 5909,
    }

    xs = {
        "A0": [],
        "A1": [],
        "B0": [],
        "B1": [],
    }
    y_tops = {
        "A0": 0.5,
        "A1": 0.6,
        "B0": 0.7,
        "B1": 0.8,
    }
    y_bottoms = {
        "A0": 0.5,
        "A1": 0.6,
        "B0": 0.7,
        "B1": 0.8,
    }

    for name, pos in lines.items():
        series = name[0:2]
        y_top = y_tops[series]
        x_ev = (ZEKE_ADIABATIC_IE_CM + pos) * CM2eV
        xs[series].append(x_ev)
        lttr = name[2]
        ax.text(x_ev, y_top, lttr, va='bottom', ha='center')

    tick_height = 0.015
    for series in ["A0", "A1", "B0", "B1"]:
        if series[0] == "A":
            color = 'tab:blue'
        elif series[0] == "B":
            color = 'tab:orange'

        y_bottom = y_bottoms[series]
        ax.vlines(xs[series], y_bottom-tick_height, y_bottom, color=color,
                  linestyles='solid', linewidths=1)

        x_min = xs[series][0]
        x_max = xs[series][-1]
        ax.hlines(y_bottom, x_min, x_max, color=color,
                  linestyles='solid', linewidths=1)

        # Add the series header
        name = series[0]
        if series[0] == "A":
            name += r"$_1$("
        else:
            name += r"$_2$("
        name += series[1] + "n0)"
        header_x_offset = -0.01
        ax.text(x_min + header_x_offset, y_bottom, name,
                va='center', ha='right')


def prepare_filename(args, config):

    # User can say the name
    user_filename = None
    if 'filename' in config:
        user_filename = config['filename']
    if args.filename is not None:
        user_filename = args.filename

    if user_filename is not None:
        return user_filename

    # If the user does not say the name create one yourself
    path = os.path.expanduser('~')
    if args.molecule is not None:
        if args.molecule == "ozone":
            path += '/ozone/plotter/pics'
        elif args.molecule == "pyrazine":
            path += '/pyrazine/calculations/absorption-spectra/xsim/pics'

    filename = path + "/"
    for idx, outname in enumerate(args.output_files):
        if idx > 0:
            filename += "+"
        filename += os.path.basename(outname)
    filename += '_spectrum.pdf'

    return filename


def find_gamma(args, config):
    gamma = None
    if 'gamma' in config:
        gamma = config['gamma']

    if args.molecule is not None and args.molecule == "pyrazine":
        gamma = 0.001

    # Command line argument takes precedence
    if args.gamma is not None:
        gamma = args.gamma

    # default value
    if gamma is None:
        gamma = 0.03

    return gamma


def get_origin(xsim_outputs):
    origin = None
    for xsim_output in xsim_outputs:
        min_element = min(xsim_output, key=lambda x: x['Energy (eV)'])
        loc_origin = min_element['Energy (eV)']
        if origin is None:
            origin = loc_origin
        elif origin > loc_origin:
            origin = loc_origin

    return origin


def add_minor_ticks(args, config, ax):
    interval = None
    if 'horizonal_minor_ticks' in config:
        interval = config['horizonal_minor_ticks']
    if args.horizonal_minor_ticks is not None:
        interval = args.horizonal_minor_ticks

    if interval is None:
        return

    ax.xaxis.set_minor_locator(MultipleLocator(0.05))


def add_second_axis(args, config, ax, origin_eV):

    second_axis = None

    if 'second_axis' in config:
        second_axis = config['second_axis']
    if args.second_axis is not None:
        second_axis = args.second_axis

    if second_axis is None:
        return

    if second_axis == "cm offset":
        add_cm_scale(ax, args, config, origin_eV)

    elif second_axis == "cm":
        add_cm_scale(ax, args, config)

    elif second_axis == "nm":
        add_nm_scale(args, ax)


def add_cm_scale(ax, args, config, origin_eV: float = None):
    ax_cm = ax.twiny()

    if origin_eV is None:
        origin_cm = 0
    else:
        origin_cm = origin_eV * eV2CM

    x1, x2 = ax.get_xlim()
    x1 = x1 * eV2CM - origin_cm
    x2 = x2 * eV2CM - origin_cm

    ax_cm.set_xlim(x1, x2)

    interval = None
    if 'horizonal_minor_ticks_2nd_axis' in config:
        interval = config['horizonal_minor_ticks_2nd_axis']
    if args.horizonal_minor_ticks_2nd_axis is not None:
        interval = args.horizonal_minor_ticks_2nd_axis

    if interval is not None:
        ax_cm.xaxis.set_minor_locator(MultipleLocator(interval))

    if args.molecule == "ozone":
        ax_cm.xaxis.set_minor_locator(MultipleLocator(200))
        # ax.xaxis.set_minor_locator(MultipleLocator(0.05))

    elif args.molecule == "ozone_zeke":
        ax_cm.xaxis.set_ticks([101100 + 300 * i for i in range(11)])
        ax_cm.xaxis.set_minor_locator(MultipleLocator(150))
        ax.xaxis.set_minor_locator(MultipleLocator(0.01))

    elif args.molecule == "ozone_dyke":
        # ax_cm.xaxis.set_ticks([101100 + 300 * i for i in range(11)])
        ax_cm.xaxis.set_minor_locator(MultipleLocator(200))
        ax.xaxis.set_ticks(np.linspace(start=12.5, stop=13.25, num=4))

    elif args.molecule == "ozone_no_cpl":
        ax_cm.xaxis.set_minor_locator(MultipleLocator(200))
        ax.xaxis.set_ticks(np.linspace(start=12.5, stop=13.25, num=4))

    # elif args.molecule == "pyrazine":
    #     ax_cm.xaxis.set_minor_locator(MultipleLocator(100))
    #     ax.xaxis.set_minor_locator(MultipleLocator(0.01))


def add_nm_scale(args, ax):
    r"""
    Relation between photon's energy and wavelength:
        E = hc / \lambda
    """
    ToLambda = 1239.84198  # from eV to nm
    ax_nm = ax.twiny()

    x1, x2 = ax.get_xlim()
    x1 = ToLambda / x1
    x2 = ToLambda / x2

    ax_nm.set_xlim(x1, x2)
    if args.molecule == "caoph":
        ax_nm.xaxis.set_minor_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(0.01))

    # if args.molecule == "ozone":
    #     ax.xaxis.set_minor_locator(MultipleLocator(0.05))


def get_fig_and_ax(args, config):

    # Scale factor is used to resize the figure in both direction
    # Making the figures smaller is the same as making the text larger
    scale_factor = 1.0
    if 'scale_factor' in config:
        scale_factor = config['scale_factor']
    if args.scale_factor is not None:
        scale_factor = args.scale_factor

    # Aspect ratio = width/height
    # width = height * ar
    aspect_ratio = 1.0
    if 'aspect_ratio' in config:
        aspect_ratio = config['aspect_ratio']

    # The default figure size is 12cm x 12 cm. Smaller should be better.
    FIGSIZE = 12 * CM2INCH * scale_factor
    fig, ax = plt.subplots(figsize=(FIGSIZE * aspect_ratio, FIGSIZE))
    return fig, ax


def add_assignmnets(args, ax, top_feature):
    if args.molecule is not None:
        if args.molecule == "ozone":
            add_ZEKE_lines(ax, top_feature)

        # elif args.molecule == "ozone_zeke":
        #     # add_ZEKE_lines(ax, 1.1)
        #     add_assignmnet_to_ZEKE_lines(ax, 1.0)
        elif args.molecule == "ozone_no_cpl":
            # add_ZEKE_lines(ax, top_feature)
            add_no_cpl_lines(ax)

#         elif args.molecule == "ozone_dyke":
#             add_ZEKE_lines(ax, 0.47)  # for the no couplings case

        elif args.molecule == "pyrazine":
            add_pyrazine_lines(ax, top_feature)

        elif args.molecule == "caoph":
            add_caoph_lines(ax, top_feature)


def set_limits(args, ax, xlims):

    ax.set_xlim([xlims[0], xlims[1]])

    # TODO: HACK:

    scale = 1e3
    ticks = ticker.FuncFormatter(
        lambda x, pos: '{0:g}'.format(x*scale))
    ax.yaxis.set_major_formatter(ticks)

    if args.molecule is not None:
        if args.molecule == "pyrazine":
            # ax.set_ylim([0.0, 0.044])
            # Only the first band
            # ax.set_ylim([0.0, 0.005])
            ax.set_ylim([0.0, 0.0045])  # The best one
            # ax.set_ylim([0.0, 0.0040])
            # The offset example
            # ax.set_ylim([0.0, 0.027])

        elif args.molecule == "ozone":
            ax.set_xlim([12.4, 13.3])
            # ax.set_ylim([0.0, 0.535])  # the best one
            ax.set_ylim([0.0, 0.55])  # the zoomed-in one

        elif args.molecule == "ozone_zeke":
            ax.set_xlim([12.51, 12.91])
            # ax.set_ylim([0.0, 0.535])  # the best one
            # ax.set_ylim([-0.1, 1.5])  # the zoomed-in one
            ax.set_ylim([-0.1, 5.1])  # Add assignmnet

        elif args.molecule == "ozone_dyke":
            ax.set_xlim([12.25, 13.40])
            # ax.set_ylim([0.0, 0.535])  # the best one
            ax.set_ylim([0.0, 0.55])  # the zoomed-in one

        elif args.molecule == "ozone_no_cpl":
            ax.set_xlim([12.25, 13.3])
            # ax.set_ylim([0.0, 0.535])  # the best one
            ax.set_ylim([0.0, 0.95])  # the zoomed-in one

        elif args.molecule == "caoph":
            ax.set_xlim([1.95, 2.35])
            # ax.set_ylim([0.0, 0.535])


def get_config(args):
    config_file = os.path.expanduser(args.config)
    if os.path.isfile(config_file) is False:
        print(f"Info: No config file {args.config} present.", file=sys.stderr)
        return {}

    print(f"Info: Using the config file {args.config}.", file=sys.stderr)
    with open(config_file, 'rb') as config_toml:
        config = tomllib.load(config_toml)

    return config


def customize_yaxsim_ticks(args, config, ax):
    show_yaxis_ticks = False
    if 'show_yaxis_ticks' in config:
        show_yaxis_ticks = config['show_yaxis_ticks']
    if args.show_yaxis_ticks is True:
        show_yaxis_ticks = True
    if show_yaxis_ticks is False:
        ax.get_yaxis().set_ticks([])


def main():
    args = parse_command_line()
    config = get_config(args)

    if args.no_parser is True or 'fort20' in config:
        basis = None
        xsim_outputs, lanczos = get_xsim_outputs_from_fort20(args)
    else:
        xsim_outputs, basis, lanczos = get_xsim_outputs(args)

    shift_eV = find_shift(xsim_outputs, args, config)
    *xlims, gamma = find_left_right_gamma(xsim_outputs, args, config)
    xsim_outputs = apply_shift(xsim_outputs, shift_eV)
    fig, ax = get_fig_and_ax(args, config)

    envelope_max_y = add_envelope(ax, args, config, xsim_outputs, xlims, gamma)
    max_peak = add_peaks(ax, args, config, xsim_outputs, xlims)
    add_info_text(ax, args, config, shift_eV, basis, lanczos, gamma)

    top_feature = max(envelope_max_y, max_peak)
    add_assignmnets(args, ax, top_feature)
    # if args.molecule == "ozone_zeke":
    #     ci_ozone_cation_cm = 104024.87948
    #     ci_ozone_cation_eV = ci_ozone_cation_cm * CM2eV
    #     # text_kw = dict(va='bottom', ha='right', rotation=-20,
    #     #                rotation_mode='anchor')
    #     xs = ci_ozone_cation_eV + shift_eV
    #     ax.vlines(xs, 0.0, 2.0, color='gray', linestyles='solid',
    #               linewidths=1, alpha=0.2)

    set_limits(args, ax, xlims)

    origin = get_origin(xsim_outputs)
    add_minor_ticks(args, config, ax)
    add_second_axis(args, config, ax, origin)

    customize_yaxsim_ticks(args, config, ax)

    fig.tight_layout()

    filename = prepare_filename(args, config)
    dont_save = False
    if 'dont_save' in config:
        dont_save = config['dont_save']
    if args.dont_save is True:
        dont_save = True

    if dont_save is True:
        plt.show()
    else:
        print(f"File saved as: {filename}")
        plt.savefig(filename)

    return 0


if __name__ == '__main__':
    main()
