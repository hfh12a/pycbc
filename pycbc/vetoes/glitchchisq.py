# Copyright (C) 2012  Alex Nitz
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
import numpy, logging, math, pycbc.fft

from pycbc.types import zeros, real_same_precision_as, TimeSeries, complex_same_precision_as
from pycbc.filter import sigmasq_series, make_frequency_series, matched_filter_core, get_cutoff_indices
from pycbc.scheme import schemed
from pycbc.vetoes.chisq import power_chisq_bins_from_sigmasq_series, power_chisq_bins
import pycbc.pnutils

BACKEND_PREFIX="pycbc.vetoes.glitchchisq_"

@schemed(BACKEND_PREFIX)
def chisq_accum_bin(chisq, q):
    pass

@schemed(BACKEND_PREFIX)
def shift_sum_max(v1, shifts, bins): #changed name
    """ Calculate the maximum value bin
    """
    pass

def glitchchisq_at_points_from_precomputed(corr, snr, snr_norm, bins, indices):
    """Calculate the chisq timeseries from precomputed values for only select points.

    This function calculates the chisq at each point by explicitly time shifting
    and summing each bin. No FFT is involved.

    Parameters
    ----------
    corr: FrequencySeries
        The product of the template and data in the frequency domain.
    snr: numpy.ndarray
        The unnormalized array of snr values at only the selected points in `indices`.
    snr_norm: float
        The normalization of the snr (EXPLAINME : refer to Findchirp paper?)
    bins: List of integers
        The edges of the equal power bins
    indices: Array
        The indices where we will calculate the chisq. These must be relative
        to the given `corr` series.

    Returns
    -------
    glitchchisq: Array
        An array containing only the glitchchisq at the selected points.
    """
    logging.info('doing fast point glitchchisq')
    num_bins = len(bins) - 1
    glitchchisq = shift_sum(corr, indices, bins)
    return (glitchchisq * num_bins - (snr.conj() * snr).real) * (snr_norm ** 2.0)

_q_l = None
_qtilde_l = None
_chisq_l = None

class SingleDetPowerGlitchChisq(object): #changed name
    """Class that handles precomputation and memory management for efficiently
    running the glitchchisq in a single detector inspiral analysis.
    """
    def __init__(self, num_bins=0, snr_threshold=None):
        if not (num_bins == "0" or num_bins == 0):
            self.do = True
            self.column_name = "glitchchisq"
            self.table_dof_name = "glitchchisq_dof"
            self.num_bins = num_bins
        else:
            self.do = False
        self.snr_threshold = snr_threshold
        self._bin_cache = {}

    @staticmethod
    def parse_option(row, arg):
        safe_dict = {}
        safe_dict.update(row.__dict__)
        safe_dict.update(math.__dict__)
        safe_dict.update(pycbc.pnutils.__dict__)
        return eval(arg, {"__builtins__":None}, safe_dict)

    def cached_chisq_bins(self, template, psd):
        key = (id(template.params), id(psd))
        if key not in self._bin_cache or not hasattr(psd, '_chisq_cached_key'):
            psd._chisq_cached_key = True
            num_bins = int(self.parse_option(template, self.num_bins))

            if hasattr(psd, 'sigmasq_vec') and template.approximant in psd.sigmasq_vec:
                logging.info("...Calculating fast power chisq bins")
                kmin = int(template.f_lower / psd.delta_f)
                kmax = template.end_idx
                bins = power_chisq_bins_from_sigmasq_series(
                    psd.sigmasq_vec[template.approximant], num_bins, kmin, kmax)
            else:
                logging.info("...Calculating power chisq bins")
                bins = power_chisq_bins(template, num_bins, psd, template.f_lower)
            self._bin_cache[key] = bins

        return self._bin_cache[key]

    def values(self, corr, snrv, snr_norm, psd, indices, template):
        """ Calculate the chisq bin with max power for each sample index.

        Returns
        -------
        glitchchisq: Array
            Chisq values, one for each sample index

        glitchchisq_dof: Array
            Number of statistical degrees of freedom for the chisq test
            in the given template
        """

        if self.do:
            logging.info("...Doing glitchchisq")

            num_above = len(indices)
            if self.snr_threshold:
                above = abs(snrv * snr_norm) > self.snr_threshold
                num_above = above.sum()
                logging.info('%s above chisq activation threshold' % num_above)
                above_indices = indices[above]
                above_snrv = snrv[above]
                rchisq = numpy.zeros(len(indices), dtype=numpy.float32)
                dof = -100
            else:
                above_indices = indices
                above_snrv = snrv

            if num_above > 0:
                bins = self.cached_chisq_bins(template, psd)
                dof = (len(bins) - 1) * 2 - 2
                glitchchisq = glitchchisq_at_points_from_precomputed(corr,
                                     above_snrv, snr_norm, bins, above_indices)

            if self.snr_threshold:
                if num_above > 0:
                    rchisq[above] = glitchchisq
            else:
                rchisq = glitchchisq

            return rchisq, numpy.repeat(dof, len(indices))# dof * numpy.ones_like(indices)
        else:
            return None, None

