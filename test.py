from madflow.config import (
    int_me,
    float_me,
    DTYPE,
    DTYPEINT,
    run_eager,
    complex_tf,
    complex_me
)
from madflow.wavefunctions_flow import oxxxxx, ixxxxx, vxxxxx, sxxxxx, _vx_no_BRST_check
from madflow.parameters import Model

import os
import sys
import numpy as np

import tensorflow as tf
import tensorflow.math as tfmath
import collections
"""
ModelParamTupleConst = collections.namedtuple("constants", ["mdl_MT","mdl_WT"])
ModelParamTupleFunc = collections.namedtuple("functions", ["GC_10","GC_11"])

root_path = '/home/gabriele/Scaricati/MG5_aMC_v3_1_1'
sys.path.insert(0, root_path)
sys.path.insert(0, os.path.join(root_path, 'madgraph'))
sys.path.insert(0, os.path.join(root_path, 'aloha', 'template_files'))

import models.import_ufo as import_ufo
import models.check_param_card as param_card_reader
"""
# import the ALOHA routines
from aloha_1_gg_ttx import *


def get_model_param(model, param_card_path):
    param_card = param_card_reader.ParamCard(param_card_path)
        # External (param_card) parameters
    aEWM1 = param_card['SMINPUTS'].get(1).value
    mdl_Gf = param_card['SMINPUTS'].get(2).value
    aS = param_card['SMINPUTS'].get(3).value
    mdl_ymb = param_card['YUKAWA'].get(5).value
    mdl_ymt = param_card['YUKAWA'].get(6).value
    mdl_ymtau = param_card['YUKAWA'].get(15).value
    mdl_MZ = param_card['MASS'].get(23).value
    mdl_MT = param_card['MASS'].get(6).value
    mdl_MB = param_card['MASS'].get(5).value
    mdl_MH = param_card['MASS'].get(25).value
    mdl_MTA = param_card['MASS'].get(15).value
    mdl_WZ = param_card['DECAY'].get(23).value
    mdl_WW = param_card['DECAY'].get(24).value
    mdl_WT = param_card['DECAY'].get(6).value
    mdl_WH = param_card['DECAY'].get(25).value

    #PS-independent parameters
    mdl_conjg__CKM1x1 = 1.0
    mdl_conjg__CKM3x3 = 1.0
    mdl_CKM3x3 = 1.0
    ZERO = 0.0
    mdl_complexi = complex(0,1)
    mdl_MZ__exp__2 = mdl_MZ**2
    mdl_MZ__exp__4 = mdl_MZ**4
    mdl_sqrt__2 =  np.sqrt(2) 
    mdl_MH__exp__2 = mdl_MH**2
    mdl_aEW = 1/aEWM1
    mdl_MW = np.sqrt(mdl_MZ__exp__2/2. + np.sqrt(mdl_MZ__exp__4/4. - (mdl_aEW*np.pi*mdl_MZ__exp__2)/(mdl_Gf*mdl_sqrt__2)))
    mdl_sqrt__aEW =  np.sqrt(mdl_aEW) 
    mdl_ee = 2*mdl_sqrt__aEW*np.sqrt(np.pi)
    mdl_MW__exp__2 = mdl_MW**2
    mdl_sw2 = 1 - mdl_MW__exp__2/mdl_MZ__exp__2
    mdl_cw = np.sqrt(1 - mdl_sw2)
    mdl_sqrt__sw2 =  np.sqrt(mdl_sw2) 
    mdl_sw = mdl_sqrt__sw2
    mdl_g1 = mdl_ee/mdl_cw
    mdl_gw = mdl_ee/mdl_sw
    mdl_vev = (2*mdl_MW*mdl_sw)/mdl_ee
    mdl_vev__exp__2 = mdl_vev**2
    mdl_lam = mdl_MH__exp__2/(2.*mdl_vev__exp__2)
    mdl_yb = (mdl_ymb*mdl_sqrt__2)/mdl_vev
    mdl_yt = (mdl_ymt*mdl_sqrt__2)/mdl_vev
    mdl_ytau = (mdl_ymtau*mdl_sqrt__2)/mdl_vev
    mdl_muH = np.sqrt(mdl_lam*mdl_vev__exp__2)
    mdl_I1x33 = mdl_yb*mdl_conjg__CKM3x3
    mdl_I2x33 = mdl_yt*mdl_conjg__CKM3x3
    mdl_I3x33 = mdl_CKM3x3*mdl_yt
    mdl_I4x33 = mdl_CKM3x3*mdl_yb
    mdl_ee__exp__2 = mdl_ee**2
    mdl_sw__exp__2 = mdl_sw**2
    mdl_cw__exp__2 = mdl_cw**2

    #PS-dependent parameters
    mdl_sqrt__aS =  np.sqrt(aS) 
    mdl_G__exp__2 = lambda G: complex_me(G**2)


    # PS-dependent couplings
    GC_10 = lambda G: complex_me(-G)
    GC_11 = lambda G: complex_me(mdl_complexi*G)

    constants = ModelParamTupleConst(float_me(mdl_MT),float_me(mdl_WT))
    functions = ModelParamTupleFunc(GC_10,GC_11)
    return Model(constants, functions)

def areclose(t1, t2):
    dist = tf.fill(tf.shape(t1), float_me(0.0000001))
    #dist = tf.fill(tf.shape(t1), float_me(0.001))
    
    # Check if the real parts are close
    
    result = tf.math.less_equal(tf.math.abs(tf.math.real(t1 - t2)), dist)
    
    tf.debugging.assert_equal(result, True)
    
    # Check if the imaginary parts are close
    
    result = tf.math.less_equal(tf.math.abs(tf.math.imag(t1 - t2)), dist)
    
    tf.debugging.assert_equal(result, True)
    print("...ok")

def vxxxxxtest(all_ps, ZERO, hel, fl, MatrixOp):
    print("Testing vxxxxx...")
    pw0 = vxxxxx(all_ps[:,0], ZERO, hel[0], fl)
    cw0 = MatrixOp.vxxxxx(all_ps, ZERO, hel[0], fl, pw0)
    areclose(pw0, cw0)
    
def oxxxxxtest(all_ps, ZERO, hel, fl, MatrixOp):
    print("Testing oxxxxx...")
    pw0 = oxxxxx(all_ps[:,2], ZERO, hel[2], fl)
    cw0 = MatrixOp.oxxxxx(all_ps, ZERO, hel[2], fl, pw0)
    areclose(pw0, cw0)
    
def ixxxxxtest(all_ps, ZERO, hel, fl, MatrixOp):
    print("Testing ixxxxx...")
    pw0 = ixxxxx(all_ps[:,3], ZERO, hel[3], fl)
    cw0 = MatrixOp.ixxxxx(all_ps, ZERO, hel[3], fl, pw0)
    #print(pw0, cw0)
    areclose(pw0, cw0)


def vxnobrstchecktest(all_ps, hel, MatrixOp):
    print("Testing _vx_no_BRST_check...")
    
    nhel = hel[0]
    p = all_ps[:,0]
    nsv = float_me(-1)
    pt2 = p[:, 1] ** 2 + p[:, 2] ** 2
    pp = tfmath.minimum(p[:, 0], tfmath.sqrt(pt2 + p[:, 3] ** 2))
    pt = tfmath.minimum(pp, tfmath.sqrt(pt2))

    hel0 = 1 - tfmath.abs(nhel)
    nsvahl = nsv * tfmath.abs(nhel)

    pw0 = _vx_no_BRST_check(p, ZERO, nhel, nsv, hel0, nsvahl, pp, pt)
    cw0 = MatrixOp.vxnobrstcheck(all_ps, ZERO, nhel, nsv, hel0, nsvahl, pp, pt, pw0)
    areclose(pw0, cw0)

def vvv1p0_1test(all_ps, hel, mdl_MT, GC_10, MatrixOp):
    print("Testing VVV1P0_1...")
    
    ZERO = float_me(0.)
    w0 = vxxxxx(all_ps[:,0],ZERO,hel[0],float_me(-1))
    w1 = vxxxxx(all_ps[:,1],ZERO,hel[1],float_me(-1))
    w2 = oxxxxx(all_ps[:,2],mdl_MT,hel[2],float_me(+1))
    w3 = ixxxxx(all_ps[:,3],mdl_MT,hel[3],float_me(-1))
    pw4= VVV1P0_1(w0, w1, GC_10, ZERO, ZERO)
    cw4 = MatrixOp.vvv1p01(all_ps, hel, w0, w1, GC_10, ZERO, ZERO, mdl_MT, pw4)
    areclose(pw4, cw4)

def ffv1_0test(all_ps, hel, mdl_MT, GC_10, GC_11, MatrixOp):
    print("Testing FFV1_0...")
    
    ZERO = float_me(0.)
    w0 = vxxxxx(all_ps[:,0],ZERO,hel[0],float_me(-1))
    w1 = vxxxxx(all_ps[:,1],ZERO,hel[1],float_me(-1))
    w2 = oxxxxx(all_ps[:,2],mdl_MT,hel[2],float_me(+1))
    w3 = ixxxxx(all_ps[:,3],mdl_MT,hel[3],float_me(-1))
    w4= VVV1P0_1(w0, w1, GC_10, ZERO, ZERO)
    
    pamp0 = FFV1_0(w3,w2,w4,GC_11)
    camp0 = MatrixOp.ffv10(all_ps, hel, w3, w2, w4, GC_10, GC_11, mdl_MT, pamp0)
    
    print(pamp0, camp0)
    areclose(pamp0, camp0)
    
if __name__ == "__main__":
    import sys, pathlib
    import numpy as np

    """
    # Read up the model
    model_sm = pathlib.Path(root_path) / "models/sm"
    if not model_sm.exists():
        print(f"No model sm found at {model_sm}, test cannot continue")
        sys.exit(0)
    model = import_ufo.import_model(model_sm.as_posix())
    model_params = get_model_param(model, 'Cards/param_card.dat')
    """
    # Define th phase space
    # The structure asked by the matrix elements is
    #   (nevents, ndimensions, nparticles)
    # the 4 dimensions of the 4-momentum is expected as
    #   (E, px, py, pz)
    ndim = 4
    npar = 4
    nevents = 2
    max_momentum = 7e3

    par_ax = 1
    dim_ax = 2

    # Now generate random outgoing particles in a com frame (last_p carries whatever momentum is needed to sum 0 )
    shape = [nevents, 0, 0]
    shape[par_ax] = npar - 3
    shape[dim_ax] = ndim - 1
    partial_out_p = tf.random.uniform(shape, minval=-max_momentum, maxval=max_momentum, dtype=DTYPE)
    last_p = -tf.reduce_sum(partial_out_p, keepdims=True, axis=par_ax)
    out_p = tf.concat([partial_out_p, last_p], axis=par_ax)
    """
    if "mdl_MT" in dir(model_params):
        # TODO fill in the mass according to the particles
        out_m = tf.reshape((npar - 2) * [model_params.mdl_MT], (1, -1, 1))
    else:
        out_m = 0.0
    out_e = tf.sqrt(tf.reduce_sum(out_p ** 2, keepdims=True, axis=dim_ax) + out_m ** 2)
    outgoing_4m = tf.concat([out_e, out_p], axis=dim_ax)

    # Put all incoming momenta in the z axis (TODO: for now assume massless input)
    ea = tf.reduce_sum(out_e, axis=par_ax, keepdims=True) / 2
    zeros = tf.zeros_like(ea)
    inc_p1 = tf.concat([ea, zeros, zeros, ea], axis=dim_ax)
    inc_p2 = tf.concat([ea, zeros, zeros, -ea], axis=dim_ax)

    all_ps = tf.concat([inc_p1, inc_p2, outgoing_4m], axis=par_ax)
    """
    hel = float_me([-1,-1,-1,1])
    ZERO = float_me(0.)
    
    # Test functions
    
    MatrixOp = tf.load_op_library('./matrix.so')
    
    
    """
    helicities = float_me([ \
        [-1,-1,-1,1],
        [-1,-1,-1,-1],
        [-1,-1,1,1],
        [-1,-1,1,-1],
        [-1,1,-1,1],
        [-1,1,-1,-1],
        [-1,1,1,1],
        [-1,1,1,-1],
        [1,-1,-1,1],
        [1,-1,-1,-1],
        [1,-1,1,1],
        [1,-1,1,-1],
        [1,1,-1,1],
        [1,1,-1,-1],
        [1,1,1,1],
        [1,1,1,-1]])
    """
    #for hel in helicities:
        #print(hel, hel[0])
    
    # 1     , 2     , 3    , 4
    # mdl_MT, mdl_WT, GC_10, GC_11
    
    mdl_MT = float_me(173.0)
    mdl_WT = float_me(1.4915000200271606)
    GC_10  = tf.constant([-1.21771579-0.j])
    GC_11  = tf.constant([0.+1.21771579j])
    
    all_ps = tf.constant([
       [[6072.61964028,    0.,            0.,         6072.61964028],
        [6072.61964028,    0.,            0.,        -6072.61964028],
        [6072.61964028,-2777.31637198, 1722.92077616, 5118.08236202],
        [6072.61964028, 2777.31637198,-1722.92077616,-5118.08236202]],
       [[3068.10329143,    0.,            0.,         3068.10329143],
        [3068.10329143,    0.,            0.,        -3068.10329143],
        [3068.10329143,-1224.19479866, 2778.9848378,  -438.00476374],
        [3068.10329143, 1224.19479866,-2778.9848378,   438.00476374]]], dtype=tf.float64)
    
    #print(all_ps)
    
    #print(mdl_MT, mdl_WT, GC_10, GC_11)
    
    #vxnobrstchecktest(all_ps, hel, MatrixOp)
    vxxxxxtest(all_ps, ZERO, hel, float_me(-1), MatrixOp)
    oxxxxxtest(all_ps, mdl_MT, hel, float_me(+1), MatrixOp)
    ixxxxxtest(all_ps, mdl_MT, hel, float_me(-1), MatrixOp)
    vvv1p0_1test(all_ps, hel, mdl_MT, GC_10, MatrixOp)
    ffv1_0test(all_ps, hel, mdl_MT, GC_10, GC_11, MatrixOp)
    #print(all_ps[:,0], all_ps[:,1])
    
    """
    model_params.freeze_alpha_s(0.118)
    """
