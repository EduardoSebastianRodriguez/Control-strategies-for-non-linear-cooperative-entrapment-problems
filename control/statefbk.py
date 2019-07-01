# statefbk.py - tools for state feedback control
#
# Author: Richard M. Murray, Roberto Bucher
# Date: 31 May 2010
#
# This file contains routines for designing state space controllers
#
# Copyright (c) 2010 by California Institute of Technology
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the California Institute of Technology nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
# OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#
# $Id$

# External packages and modules
import numpy as np
import scipy as sp
from . import statesp
from .exception import ControlSlycot, ControlArgument, ControlDimension

__all__ = ['ctrb', 'obsv', 'gram', 'place', 'place_varga', 'lqr', 'acker']


# Pole placement
def place(A, B, p):
    """Place closed loop eigenvalues
    K = place(A, B, p)

    Parameters
    ----------
    A : 2-d array
        Dynamics matrix
    B : 2-d array
        Input matrix
    p : 1-d list
        Desired eigenvalue locations

    Returns
    -------
    K : 2-d array
        Gain such that A - B K has eigenvalues given in p

    Algorithm
    ---------
    This is a wrapper function for scipy.signal.place_poles, which
    implements the Tits and Yang algorithm [1]. It will handle SISO,
    MISO, and MIMO systems. If you want more control over the algorithm,
    use scipy.signal.place_poles directly.

    [1] A.L. Tits and Y. Yang, "Globally convergent algorithms for robust
    pole assignment by state feedback, IEEE Transactions on Automatic
    Control, Vol. 41, pp. 1432-1452, 1996.

    Limitations
    -----------
    The algorithm will not place poles at the same location more
    than rank(B) times.

    Examples
    --------
    >>> A = [[-1, -1], [0, 1]]
    >>> B = [[0], [1]]
    >>> K = place(A, B, [-2, -5])

    See Also
    --------
    place_varga, acker
    """
    from scipy.signal import place_poles

    # Convert the system inputs to NumPy arrays
    A_mat = np.array(A)
    B_mat = np.array(B)
    if (A_mat.shape[0] != A_mat.shape[1]):
        raise ControlDimension("A must be a square matrix")

    if (A_mat.shape[0] != B_mat.shape[0]):
        err_str = "The number of rows of A must equal the number of rows in B"
        raise ControlDimension(err_str)

    # Convert desired poles to numpy array
    placed_eigs = np.array(p)

    result = place_poles(A_mat, B_mat, placed_eigs, method='YT')
    K = result.gain_matrix
    return K


def place_varga(A, B, p):
    """Place closed loop eigenvalues
    K = place_varga(A, B, p)

    Parameters
    ----------
    A : 2-d array
        Dynamics matrix
    B : 2-d array
        Input matrix
    p : 1-d list
        Desired eigenvalue locations
    Returns
    -------
    K : 2-d array
        Gain such that A - B K has eigenvalues given in p.


    Algorithm
    ---------
        This function is a wrapper for the slycot function sb01bd, which
        implements the pole placement algorithm of Varga [1]. In contrast to
        the algorithm used by place(), the Varga algorithm can place
        multiple poles at the same location. The placement, however, may not
        be as robust.

        [1] Varga A. "A Schur method for pole assignment."
            IEEE Trans. Automatic Control, Vol. AC-26, pp. 517-519, 1981.

    Examples
    --------
    >>> A = [[-1, -1], [0, 1]]
    >>> B = [[0], [1]]
    >>> K = place(A, B, [-2, -5])

    See Also:
    --------
    place, acker
    """

    # Make sure that SLICOT is installed
    try:
        from slycot import sb01bd
    except ImportError:
        raise ControlSlycot("can't find slycot module 'sb01bd'")

    # Convert the system inputs to NumPy arrays
    A_mat = np.array(A);
    B_mat = np.array(B);
    if (A_mat.shape[0] != A_mat.shape[1] or
        A_mat.shape[0] != B_mat.shape[0]):
        raise ControlDimension("matrix dimensions are incorrect")

    # Compute the system eigenvalues and convert poles to numpy array
    system_eigs = np.linalg.eig(A_mat)[0]
    placed_eigs = np.array(p);

    # SB01BD sets eigenvalues with real part less than alpha
    # We want to place all poles of the system => set alpha to minimum
    alpha = min(system_eigs.real);

    # Call SLICOT routine to place the eigenvalues
    A_z,w,nfp,nap,nup,F,Z = \
        sb01bd(B_mat.shape[0], B_mat.shape[1], len(placed_eigs), alpha,
               A_mat, B_mat, placed_eigs, 'C');

    # Return the gain matrix, with MATLAB gain convention
    return -F

# Contributed by Roberto Bucher <roberto.bucher@supsi.ch>
def acker(A, B, poles):
    """Pole placement using Ackermann method

    Call:
    K = acker(A, B, poles)

    Parameters
    ----------
    A, B : 2-d arrays
        State and input matrix of the system
    poles: 1-d list
        Desired eigenvalue locations

    Returns
    -------
    K: matrix
        Gains such that A - B K has given eigenvalues

    """
    # Convert the inputs to matrices
    a = np.mat(A)
    b = np.mat(B)

    # Make sure the system is controllable
    ct = ctrb(A, B)
    if np.linalg.matrix_rank(ct) != a.shape[0]:
        raise ValueError("System not reachable; pole placement invalid")

    # Compute the desired characteristic polynomial
    p = np.real(np.poly(poles))

    # Place the poles using Ackermann's method
    n = np.size(p)
    pmat = p[n-1]*a**0
    for i in np.arange(1,n):
        pmat = pmat + p[n-i-1]*a**i
    K = np.linalg.solve(ct, pmat)

    K = K[-1][:]                # Extract the last row
    return K

def lqr(*args, **keywords):
    """lqr(A, B, Q, R[, N])

    Linear quadratic regulator design

    The lqr() function computes the optimal state feedback controller
    that minimizes the quadratic cost

    .. math:: J = \int_0^\infty (x' Q x + u' R u + 2 x' N u) dt

    The function can be called with either 3, 4, or 5 arguments:

    * ``lqr(sys, Q, R)``
    * ``lqr(sys, Q, R, N)``
    * ``lqr(A, B, Q, R)``
    * ``lqr(A, B, Q, R, N)``

    where `sys` is an `LTI` object, and `A`, `B`, `Q`, `R`, and `N` are
    2d arrays or matrices of appropriate dimension.

    Parameters
    ----------
    A, B: 2-d array
        Dynamics and input matrices
    sys: LTI (StateSpace or TransferFunction)
        Linear I/O system
    Q, R: 2-d array
        State and input weight matrices
    N: 2-d array, optional
        Cross weight matrix

    Returns
    -------
    K: 2-d array
        State feedback gains
    S: 2-d array
        Solution to Riccati equation
    E: 1-d array
        Eigenvalues of the closed loop system

    Examples
    --------
    >>> K, S, E = lqr(sys, Q, R, [N])
    >>> K, S, E = lqr(A, B, Q, R, [N])

    """

    # Make sure that SLICOT is installed
    try:
        from slycot import sb02md
        from slycot import sb02mt
    except ImportError:
        raise ControlSlycot("can't find slycot module 'sb02md' or 'sb02nt'")

    #
    # Process the arguments and figure out what inputs we received
    #

    # Get the system description
    if (len(args) < 3):
        raise ControlArgument("not enough input arguments")

    try:
        # If this works, we were (probably) passed a system as the
        # first argument; extract A and B
        A = np.array(args[0].A, ndmin=2, dtype=float);
        B = np.array(args[0].B, ndmin=2, dtype=float);
        index = 1;
    except AttributeError:
        # Arguments should be A and B matrices
        A = np.array(args[0], ndmin=2, dtype=float);
        B = np.array(args[1], ndmin=2, dtype=float);
        index = 2;

    # Get the weighting matrices (converting to matrices, if needed)
    Q = np.array(args[index], ndmin=2, dtype=float);
    R = np.array(args[index+1], ndmin=2, dtype=float);
    if (len(args) > index + 2):
        N = np.array(args[index+2], ndmin=2, dtype=float);
    else:
        N = np.zeros((Q.shape[0], R.shape[1]));

    # Check dimensions for consistency
    nstates = B.shape[0];
    ninputs = B.shape[1];
    if (A.shape[0] != nstates or A.shape[1] != nstates):
        raise ControlDimension("inconsistent system dimensions")

    elif (Q.shape[0] != nstates or Q.shape[1] != nstates or
          R.shape[0] != ninputs or R.shape[1] != ninputs or
          N.shape[0] != nstates or N.shape[1] != ninputs):
        raise ControlDimension("incorrect weighting matrix dimensions")

    # Compute the G matrix required by SB02MD
    A_b,B_b,Q_b,R_b,L_b,ipiv,oufact,G = \
        sb02mt(nstates, ninputs, B, R, A, Q, N, jobl='N');

    # Call the SLICOT function
    X,rcond,w,S,U,A_inv = sb02md(nstates, A_b, G, Q_b, 'C')

    # Now compute the return value
    # We assume that R is positive definite and, hence, invertible
    K = np.linalg.solve(R, np.dot(B.T, X) + N.T);
    S = X;
    E = w[0:nstates];

    return K, S, E

def ctrb(A,B):
    """Controllabilty matrix

    Parameters
    ----------
    A, B: array_like or string
        Dynamics and input matrix of the system

    Returns
    -------
    C: matrix
        Controllability matrix

    Examples
    --------
    >>> C = ctrb(A, B)

    """

    # Convert input parameters to matrices (if they aren't already)
    amat = np.mat(A)
    bmat = np.mat(B)
    n = np.shape(amat)[0]
    # Construct the controllability matrix
    ctrb = bmat
    for i in range(1, n):
        ctrb = np.hstack((ctrb, amat**i*bmat))
    return ctrb

def obsv(A, C):
    """Observability matrix

    Parameters
    ----------
    A, C: array_like or string
        Dynamics and output matrix of the system

    Returns
    -------
    O: matrix
        Observability matrix

    Examples
    --------
    >>> O = obsv(A, C)

   """

    # Convert input parameters to matrices (if they aren't already)
    amat = np.mat(A)
    cmat = np.mat(C)
    n = np.shape(amat)[0]

    # Construct the controllability matrix
    obsv = cmat
    for i in range(1, n):
        obsv = np.vstack((obsv, cmat*amat**i))
    return obsv

def gram(sys,type):
    """Gramian (controllability or observability)

    Parameters
    ----------
    sys: StateSpace
        State-space system to compute Gramian for
    type: String
        Type of desired computation.
        `type` is either 'c' (controllability) or 'o' (observability). To compute the
        Cholesky factors of gramians use 'cf' (controllability) or 'of' (observability)

    Returns
    -------
    gram: array
        Gramian of system

    Raises
    ------
    ValueError
        * if system is not instance of StateSpace class
        * if `type` is not 'c', 'o', 'cf' or 'of'
        * if system is unstable (sys.A has eigenvalues not in left half plane)

    ImportError
        if slycot routine sb03md cannot be found
        if slycot routine sb03od cannot be found

    Examples
    --------
    >>> Wc = gram(sys,'c')
    >>> Wo = gram(sys,'o')
    >>> Rc = gram(sys,'cf'), where Wc=Rc'*Rc
    >>> Ro = gram(sys,'of'), where Wo=Ro'*Ro

    """

    #Check for ss system object
    if not isinstance(sys,statesp.StateSpace):
        raise ValueError("System must be StateSpace!")
    if type not in ['c', 'o', 'cf', 'of']:
        raise ValueError("That type is not supported!")

    #TODO: Check for continous or discrete, only continuous supported right now
        # if isCont():
        #    dico = 'C'
        # elif isDisc():
        #    dico = 'D'
        # else:
    dico = 'C'

    #TODO: Check system is stable, perhaps a utility in ctrlutil.py
        # or a method of the StateSpace class?
    if np.any(np.linalg.eigvals(sys.A).real >= 0.0):
        raise ValueError("Oops, the system is unstable!")

    if type=='c' or type=='o':
        #Compute Gramian by the Slycot routine sb03md
        #make sure Slycot is installed
        try:
            from slycot import sb03md
        except ImportError:
            raise ControlSlycot("can't find slycot module 'sb03md'")
        if type=='c':
            tra = 'T'
            C = -np.dot(sys.B,sys.B.transpose())
        elif type=='o':
            tra = 'N'
            C = -np.dot(sys.C.transpose(),sys.C)
        n = sys.states
        U = np.zeros((n,n))
        A = np.array(sys.A)         # convert to NumPy array for slycot
        X,scale,sep,ferr,w = sb03md(n, C, A, U, dico, job='X', fact='N', trana=tra)
        gram = X
        return gram

    elif type=='cf' or type=='of':
        #Compute cholesky factored gramian from slycot routine sb03od
        try:
            from slycot import sb03od
        except ImportError:
            raise ControlSlycot("can't find slycot module 'sb03od'")
        tra='N'
        n = sys.states
        Q = np.zeros((n,n))
        A = np.array(sys.A)         # convert to NumPy array for slycot
        if type=='cf':
            m = sys.B.shape[1]
            B = np.zeros_like(A)
            B[0:m,0:n] = sys.B.transpose()
            X,scale,w = sb03od(n, m, A.transpose(), Q, B, dico, fact='N', trans=tra)
        elif type=='of':
            m = sys.C.shape[0]
            C = np.zeros_like(A)
            C[0:n,0:m] = sys.C.transpose()
            X,scale,w = sb03od(n, m, A, Q, C.transpose(), dico, fact='N', trans=tra)
        gram = X
        return gram