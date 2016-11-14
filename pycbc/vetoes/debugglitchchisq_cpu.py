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
import numpy, pycbc
from pycbc.types import real_same_precision_as
from scipy.weave import inline
from pycbc import WEAVE_FLAGS

if pycbc.HAVE_OMP:
    omp_libs = ['gomp']
    omp_flags = ['-fopenmp']
else:
    omp_libs = []
    omp_flags = []

complex_code = """
#include <complex.h>
"""
    
debug_glitchchisq_code = """ 
    std::complex<float> I(0.0, 1.0);
    int num_parallel_regions = 16;
    
    for (unsigned int r=0; r<nbins; r++){     
        int bstart = bins[r];
        int bend = bins[r+1];
        int blen = bend - bstart;
        
        TYPE* outr = (TYPE*) malloc(sizeof(TYPE)*n);
        TYPE* outi = (TYPE*) malloc(sizeof(TYPE)*n);
        for (int i=0; i<n; i++){
            outr[i] = 0;
            outi[i] = 0;
        }
    
        #pragma omp parallel for
        for (unsigned int k=0; k<num_parallel_regions; k++){
            unsigned int start = blen * k / num_parallel_regions + bstart;
            unsigned int end = blen * (k + 1) / num_parallel_regions + bstart;
        
            //start the cumulative rotations at the offset point
            TYPE* pr = (TYPE*) malloc(sizeof(TYPE)*n);
            TYPE* pi = (TYPE*) malloc(sizeof(TYPE)*n);
            TYPE* vsr = (TYPE*) malloc(sizeof(TYPE)*n);
            TYPE* vsi = (TYPE*) malloc(sizeof(TYPE)*n);
            TYPE* outr_tmp = (TYPE*) malloc(sizeof(TYPE)*n);
            TYPE* outi_tmp = (TYPE*) malloc(sizeof(TYPE)*n);

            
            for (int i=0; i<n; i++){
                pr[i] = cos(2 * 3.141592653 * shifts[i] * (start) / slen);
                pi[i] = sin(2 * 3.141592653 * shifts[i] * (start) / slen);
                vsr[i] = cos(2 * 3.141592653 * shifts[i] / slen);
                vsi[i] = sin(2 * 3.141592653 * shifts[i] / slen);
                outr_tmp[i] = 0;
                outi_tmp[i] = 0;
            }
            
            TYPE t1, t2, k1, k2, k3, vs, va; 
            
            for (unsigned int j=start; j<end; j++){
                std::complex<TYPE> v = v1[j];
                TYPE vr = v.real();
                TYPE vi = v.imag();  
                vs = vr + vi;
                va = vi - vr;
                
                for (int i=0; i<n; i++){
                    t1 = pr[i];
                    t2 = pi[i];
                    
                    // Complex multiply pr[i] * v
                    k1 = vr * (t1 + t2);
                    k2 = t1 * va;
                    k3 = t2 * vs;
                                
                    outr_tmp[i] += k1 - k3;
                    outi_tmp[i] += k1 + k2;
                    
                    // phase shift for the next time point
                    pr[i] = t1 * vsr[i] - t2 * vsi[i];
                    pi[i] = t1 * vsi[i] + t2 * vsr[i]; 
                }                                              
            } 
            
            #pragma omp critical
            {
                for (unsigned int i=0; i<n; i++){
                    outr[i] += outr_tmp[i];
                    outi[i] += outi_tmp[i];
                }
            }
            
            free(pr);
            free(pi);  
            free(outr_tmp);
            free(outi_tmp); 
            free(vsr);
            free(vsi);
            
        }

        // n is the number of triggers above threshhold
        for (unsigned int i=0; i<n; i++){
	    outarray[r*nbins + i] = outr[i] + (outi[i]* I);
        }
        free(outr);
        free(outi);
    } // End loop over number of chi-square bins    
"""

debugglitchchisq_code_single = debug_glitchchisq_code.replace('TYPE', 'float')
debugglitchchisq_code_double = debug_glitchchisq_code.replace('TYPE', 'double')

def debug_shift_sum_max(v1, shifts, bins):
    real_type = real_same_precision_as(v1)
    shifts = numpy.array(shifts, dtype=real_type)
    
    bins = numpy.array(bins, dtype=numpy.uint32)
    nbins = len(bins) - 1
    v1 = numpy.array(v1.data, copy=False)
    slen = len(v1)

    if v1.dtype.name == 'complex64':
        code = debugglitchchisq_code_single
    else:
        code = debugglitchchisq_code_double
    
    n = int(len(shifts))
    
    # Create some output memory
    debugglitchchisq =  numpy.zeros(n, dtype=real_type)
    outarray = numpy.zeros(len(bins)*nbins, dtype=numpy.complex64)
    inline(code, ['v1', 'n', 'debugglitchchisq', 'slen', 'shifts', 'bins', 'nbins', 'outarray'],
                    extra_compile_args=[WEAVE_FLAGS] + omp_flags,
                    libraries=omp_libs,
		    support_code = complex_code)
    numpy.reshape(outarray, (nbins,len(debugglitchchisq)))
    return debugglitchchisq
