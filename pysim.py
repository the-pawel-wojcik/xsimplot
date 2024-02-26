#!/usr/bin/env python

import argparse
import os
import sys
import math as m
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import numpy as np
from parsers.xsim import parse_xsim_output

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
        description="Plot output of the xsim program.")

    parser.add_argument("output_files",
                        help="List of xsim output files.",
                        nargs="+")

    parser.add_argument("-c", "--cm_scale",
                        help="Add a second energy scale in (cm-1).",
                        default=False,
                        action="store_true")

    parser.add_argument("-e", "--envelope",
                        help="Add a Lorenzian envelope to every peak.",
                        type=str,
                        default=None,
                        choices=['stack', 'overlay'])

    parser.add_argument("-r", "--scale_factor",
                        help="Scale the figure size with the factor.",
                        default=1.0,
                        type=float)

    parser.add_argument("-g", "--gamma",
                        help="Gamma in the Lorenzian:\n" +
                        "(0.5 * gamma)**2 / ((x - x0)**2 + (0.5 * gamma) **2)",
                        type=float,
                        default=0.030)

    parser.add_argument("-t", "--sticks_off",
                        help="Do NOT show stick spectrum.",
                        default=False,
                        action="store_true")

    parser.add_argument("-v", "--verbose",
                        help="Annotate with # of Lanczos it and basis size.",
                        default=False,
                        action="store_true")

    help = "Match some of the plot properties (like xlims or output location)"
    help += " to the selected molecule."
    parser.add_argument("-m", "--molecule",
                        help=help,
                        type=str,
                        choices=["ozone", "pyrazine"])

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


def get_xsim_outputs(args):
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
            print("Warning! Outputs use different basis sets.")

        loc_lanczos = xsim_data['Lanczos']
        if lanczos is None:
            lanczos = loc_lanczos
        elif lanczos != loc_lanczos:
            print("Warning! Outputs use different # of Lanczos iterations.")

    return xsim_outputs, basis, lanczos


def find_shift(xsim_outputs, args):
    """
    Find shift (in eV) which will be applied to the spectrum.
    The shift is 
    """

    if args.shift_eV is not None:
        return args.shift_eV

    first_peak_position = args.match_origin

    if args.molecule is not None:
        if args.molecule == "ozone":
            # Position of the first ionization energy of ozone
            first_peak_position = ZEKE_ADIABATIC_IE_CM * CM2eV
        elif args.molecule == "pyrazine":
            first_peak_position = PYRAZINE_ABSORPTION_ORIGIN_CM * CM2eV

    if first_peak_position is not None:
        origins = []
        for xsim_output in xsim_outputs:
            min_element = min(xsim_output, key=lambda x: x['Energy (eV)'])
            origins.append(min_element['Energy (eV)'])
        origin_eV = min(origins)
        shift_eV = origin_eV - first_peak_position
        return shift_eV

    return None


def find_left_right(xsim_outputs, args, how_far: int = 4):
    """
    Find the positions of the lowest energy and highest energy peaks and return
    values how_far*gamma away from them as the plot limits.
    """

    if args.molecule is not None:
        if args.molecule == "ozone":
            # First photoelectron band of ozone
            left = 12.225
            right = 13.375
        elif args.molecule == "pyrazine":
            # First absorption band of pyrazine
            # left = 3.75
            # right = 4.25
            left = 3.8
            right = 4.0
        return (left, right)

    mins = list()
    maxes = list()

    for xsim_output in xsim_outputs:
        min_element = min(xsim_output, key=lambda x: x['Energy (eV)'])
        max_element = max(xsim_output, key=lambda x: x['Energy (eV)'])
        mins.append(min_element['Energy (eV)'])
        maxes.append(max_element['Energy (eV)'])

    minimum = min(mins)
    maximum = max(maxes)
    gamma_eV = args.gamma
    left = minimum - how_far * gamma_eV
    right = maximum + how_far * gamma_eV

    return (left, right)


def apply_shift(xsim_outputs, shift_eV):
    if shift_eV is None:
        return xsim_outputs

    for output in xsim_outputs:
        for peak in output:
            peak['Energy (eV)'] -= shift_eV

    return xsim_outputs


def add_envelope(ax, args, xsim_outputs, left, right, gamma):
    if args.envelope is None:
        return 0.0

    npoints = 500
    xs = np.linspace(left, right, npoints)
    accumutated_ys = np.zeros_like(xs)

    for file_idx, xsim_output in enumerate(xsim_outputs):
        if file_idx == len(COLORS) or file_idx > len(COLORS):
            print("Too many colors already.", file=sys.stderr)
            sys.exit(1)

        state_spectrum = [lorenz_intensity(x, gamma, xsim_output) for x in xs]
        state_spectrum = np.array(state_spectrum)
        color = COLORS[file_idx]
        if args.envelope == "stack":
            ax.fill_between(xs, accumutated_ys + state_spectrum,
                            accumutated_ys, color=color, alpha=0.2)
        elif args.envelope == "overlay":
            ax.fill_between(xs, state_spectrum, np.zeros_like(xs), color=color,
                            alpha=0.2)
        accumutated_ys += state_spectrum

    # Plot the total spectrum extra for overlay
    if args.envelope == "overlay":
        ax.plot(xs, accumutated_ys, color='tab:gray', lw=1)

    fig_max_y = np.max(accumutated_ys)
    return fig_max_y


def add_peaks(ax, xsim_outputs, args):
    sticks_off = args.sticks_off
    if sticks_off is True:
        return 0.0

    peaks_maxima = []
    for file_idx, xsim_output in enumerate(xsim_outputs):
        if file_idx == len(COLORS) or file_idx > len(COLORS):
            print("Too many colors already.")
            sys.exit(1)
        stem_xsim_output(xsim_output, ax, COLORS[file_idx])

        max_peak = max(xsim_output, key=lambda x: x['Relative intensity'])
        peaks_maxima.append(max_peak['Relative intensity'])

    return max(peaks_maxima)


def add_info_text(ax, args, shift_eV, basis, lanczos, gamma):
    info_kwargs = {'horizontalalignment': 'left',
                   'verticalalignment': 'top',
                   'fontsize': FONTSIZE,
                   'color': 'k',
                   'transform': ax.transAxes,
                   }

    text = ""

    if args.envelope is not None:
        text = r'$\gamma = ' + f'{gamma:.3f}$\n'

    if shift_eV is not None:
        text += f'$s = {shift_eV:.2f}$'

    if args.verbose is True:
        text += f'\nBasis: {basis.split()[0]}'
        text += f'\nLanczos: {lanczos}'

    ax.text(0.01, 0.99, text, **info_kwargs)


def add_pyrazine_lines(ax, top_feature):
    PYRAZINE_LINES = [{'pos_cm': 0.0, 'lbl': '0.0', 'series': 0},
                      {'pos_cm': 383, 'lbl': r'$10a ^1$', 'series': 1},
                      # {'pos_cm': 467, 'lbl': r'$16 b^2$', 'series': 3},
                      # {'pos_cm': 517, 'lbl': r'$5 ^1$', 'series': 4},
                      {'pos_cm': 583, 'lbl': r'$6a ^1$', 'series': 2},
                      {'pos_cm': 823, 'lbl': r'$10 a^2$', 'series': 1},
                      {'pos_cm': 945, 'lbl': r'$10 a^1 6 a^1$', 'series': 3},
                      {'pos_cm': 1167, 'lbl': r'$6a ^2$', 'series': 2},]

    # top_feature = 0.038
    top_feature = 0.004
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
        series = line['series']
        x_cm = pos_cm + PYRAZINE_ABSORPTION_ORIGIN_CM
        x_eV = x_cm * CM2eV
        ax.text(x_eV, y_top - series * step, lttr, **text_kwargs)
        ax.vlines([x_eV], 0.0, [y_bottom - series * step], **line_kwargs)


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

    x = ZEKE_2ND_ADIABATIC_IE_CM
    x_ev = x * CM2eV
    # lttr = r'$\tilde{A} ^+ (0,0,0)$'
    lttr = r'$IE _2$'
    y_bottom *= 0.9
    y_top *= 0.9
    ax.text(x_ev, y_top, lttr, va='center', ha='center')
    ax.vlines(x_ev, 0.0, y_bottom, color='gray', linestyles='solid',
              linewidths=1, alpha=0.2)

    x = ZEKE_DISSOCIATION_THRESHOLD
    x_ev = x * CM2eV
    lttr = r'$D _0$'
    ax.text(x_ev, y_top, lttr, va='center', ha='center')
    ax.vlines(x_ev, 0.0, y_bottom, color='gray', linestyles='solid',
              linewidths=1, alpha=0.2)


def prepare_filename(args):
    if args.filename is not None:
        return args.filename

    chem = '/home/pawel/chemistry'
    path = chem
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
    filename += '.pdf'

    return filename


def find_gamma(args):
    gamma = args.gamma
    if args.molecule is not None and args.molecule == "pyrazine":
        gamma = 0.001
    return gamma


def get_origin(args, xsim_outputs):
    origin = None
    for xsim_output in xsim_outputs:
        min_element = min(xsim_output, key=lambda x: x['Energy (eV)'])
        loc_origin = min_element['Energy (eV)']
        if origin is None:
            origin = loc_origin
        elif origin > loc_origin:
            origin = loc_origin

    return origin


def add_cm_scale(args, ax, origin_eV):
    if args.cm_scale is False:
        return
    ax_cm = ax.twiny()

    origin_cm = origin_eV * eV2CM
    x1, x2 = ax.get_xlim()
    x1 = x1 * eV2CM - origin_cm
    x2 = x2 * eV2CM - origin_cm

    ax_cm.set_xlim(x1, x2)
    if args.molecule == "ozone":
        ax_cm.xaxis.set_minor_locator(MultipleLocator(200))

    if args.molecule == "ozone":
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))


def main():
    args = parse_command_line()
    gamma = find_gamma(args)
    xsim_outputs, basis, lanczos = get_xsim_outputs(args)
    shift_eV = find_shift(xsim_outputs, args)
    left, right = find_left_right(xsim_outputs, args)
    xsim_outputs = apply_shift(xsim_outputs, shift_eV)

    scale_factor = args.scale_factor
    FIGSIZE = 12 * CM2INCH * scale_factor
    fig, ax = plt.subplots(figsize=(FIGSIZE, FIGSIZE))
    envelope_max_y = add_envelope(ax, args, xsim_outputs, left, right, gamma)
    max_peak = add_peaks(ax, xsim_outputs, args)
    # add_info_text(ax, args, shift_eV, basis, lanczos, gamma)

    if args.molecule is not None:
        top_feature = max(envelope_max_y, max_peak)
        if args.molecule == "ozone":
            pass
            add_ZEKE_lines(ax, top_feature)
        elif args.molecule == "pyrazine":
            add_pyrazine_lines(ax, top_feature)

    ax.set_xlim([left, right])
    if args.molecule is not None:
        if args.molecule == "pyrazine":
            # ax.set_ylim([0.0, 0.044])
            ax.set_ylim([0.0, 0.007])

        if args.molecule == "ozone":
            ax.set_xlim([12.4, 13.3])
            ax.set_ylim([0.0, 0.535])

    origin = get_origin(args, xsim_outputs)
    add_cm_scale(args, ax, origin)

    # Disable the y-axis ticks
    # ax.get_yaxis().set_ticks([])
    fig.tight_layout()

    # exp_spectrum_png = plt.imread("./pics/Experiments/Dyke-et-al-1974.png")
    # ax.imshow(exp_spectrum_png, zorder=-1, extent=[left, right, 0.0,
    #                                                1.1 * fig_max_y])

    filename = prepare_filename(args)
    if args.dont_save is True:
        plt.show()
    else:
        print(f"File saved as: {filename}")
        plt.savefig(filename)

    return 0


if __name__ == '__main__':
    main()
