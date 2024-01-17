import numpy as np
sim_jacobian = np.array([[ 6.76729996e-03, -1.25500843e-01,  6.76839147e-03,
         4.18500721e-01, -6.77030347e-03,  7.59996772e-02,
         3.30601324e-09],
       [ 2.16020301e-01, -8.20081254e-07,  2.16020271e-01,
        -5.43745728e-07, -8.60200226e-02, -3.09610414e-07,
         2.59460649e-08],
       [-9.39356859e-09, -2.16020316e-01, -1.36242555e-08,
         1.63520277e-01, -6.05621040e-07,  8.60200301e-02,
        -6.77000871e-03],
       [-5.96046448e-08, -5.96046448e-08, -6.70552254e-08,
         1.27591193e-07, -5.58793545e-09,  0.00000000e+00,
         1.00000000e+00],
       [ 4.47034836e-08,  1.00000012e+00,  5.96046448e-08,
        -1.00000012e+00,  7.04079866e-06, -1.00000000e+00,
         3.68244946e-06],
       [ 1.00000000e+00, -3.69548798e-06,  1.00000000e+00,
        -3.60887498e-06, -1.00000012e+00, -3.57627869e-06,
         4.95056156e-07]], dtype=np.float32
)

real_jacobian = np.array([[ 6.7673484e-03, -1.2550084e-01,  6.7684399e-03,  4.1850072e-01,
        -6.7703351e-03,  7.5999700e-02,  1.2310011e-09],
       [ 2.1602027e-01, -8.2008114e-07,  2.1602024e-01, -5.4374561e-07,
        -8.6020000e-02, -3.0975457e-07,  2.4650944e-08],
       [-9.3935633e-09, -2.1602029e-01, -1.3624251e-08,  1.6352025e-01,
        -5.9467288e-07,  8.6020015e-02, -6.7700096e-03],
       [-5.9604645e-08, -5.9604645e-08, -6.7055225e-08,  1.2759119e-07,
        -1.3184035e-08, -2.3841858e-07,  1.0000001e+00],
       [ 4.4703484e-08,  1.0000001e+00,  5.9604645e-08, -1.0000001e+00,
         6.9141388e-06, -1.0000002e+00,  3.4738332e-06],
       [ 1.0000000e+00, -3.6954880e-06,  1.0000000e+00, -3.6088750e-06,
        -1.0000001e+00, -3.2484531e-06,  1.8160790e-07]], dtype=np.float32
)

print(sim_jacobian - real_jacobian)
print(np.allclose(sim_jacobian, real_jacobian, rtol=1e-04, atol=1e-04))