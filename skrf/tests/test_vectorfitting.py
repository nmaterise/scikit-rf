import unittest
import skrf
import numpy as np
import tempfile
import os
import warnings
import matplotlib.pyplot as mplt


"""
pytest -s -v -c "" skrf/tests/test_vectorfitting.py -k "test_passivity_enforcement_hybrid_mcdermott"
"""
import datetime


test_all = False

class VectorFittingTestCase(unittest.TestCase):

    #@pytest.mark.skipif(not test_all)
    def test_ringslot_with_proportional(self):
        # perform the fit
        nw = skrf.data.ring_slot
        vf = skrf.vectorFitting.VectorFitting(nw)
        vf.vector_fit(n_poles_real=2, n_poles_cmplx=0, fit_proportional=True, fit_constant=True)
        self.assertLess(vf.get_rms_error(), 0.02)

    #@pytest.mark.skipif(not test_all)
    def test_ringslot_default_log(self):
        # perform the fit without proportional term
        nw = skrf.data.ring_slot
        vf = skrf.vectorFitting.VectorFitting(nw)
        vf.vector_fit(n_poles_real=4, n_poles_cmplx=0, init_pole_spacing='log')
        self.assertLess(vf.get_rms_error(), 0.01)

    #@pytest.mark.skipif(not test_all)
    def test_ringslot_without_prop_const(self):
        # perform the fit without proportional term
        nw = skrf.data.ring_slot
        vf = skrf.vectorFitting.VectorFitting(nw)
        vf.vector_fit(n_poles_real=4, n_poles_cmplx=0, fit_proportional=False, fit_constant=False)
        self.assertLess(vf.get_rms_error(), 0.01)

    #@pytest.mark.skipif(not test_all)
    def test_190ghz_measured(self):
        # perform the fit without proportional term
        nw = skrf.network.Network('./doc/source/examples/vectorfitting/190ghz_tx_measured.S2P')
        vf = skrf.vectorFitting.VectorFitting(nw)
        vf.vector_fit(n_poles_real=4, n_poles_cmplx=4, fit_proportional=False, fit_constant=True)
        self.assertLess(vf.get_rms_error(), 0.02)

    #@pytest.mark.skipif(not test_all)
    def test_no_convergence(self):
        # perform a bad fit that does not converge and check if a RuntimeWarning is given
        with warnings.catch_warnings(record=True) as warning:
            nw = skrf.network.Network('./doc/source/examples/vectorfitting/190ghz_tx_measured.S2P')
            vf = skrf.vectorFitting.VectorFitting(nw)
            vf.vector_fit(n_poles_real=0, n_poles_cmplx=5, fit_proportional=False, fit_constant=True)
            self.assertEqual(warning[-1].category, RuntimeWarning)

    #@pytest.mark.skipif(not test_all)
    def test_spice_subcircuit(self):
        # fit ring slot example network
        nw = skrf.data.ring_slot
        vf = skrf.vectorFitting.VectorFitting(nw)
        vf.vector_fit(n_poles_real=4, n_poles_cmplx=0, fit_constant=True, fit_proportional=True)

        # write equivalent SPICE subcircuit to tmp file
        tmp_file = tempfile.NamedTemporaryFile(suffix='.sp')
        tmp_file.close()  # tmp_file.name can be used to open the file a second time on Linux but not on Windows
        vf.write_spice_subcircuit_s(tmp_file.name)

        # written tmp file should contain 69 lines
        n_lines = len(open(tmp_file.name, 'r').readlines())
        self.assertEqual(n_lines, 69)

    #@pytest.mark.skipif(not test_all)
    def test_read_write_npz(self):
        # fit ring slot example network
        nw = skrf.data.ring_slot
        vf = skrf.vectorFitting.VectorFitting(nw)
        vf.vector_fit(n_poles_real=3, n_poles_cmplx=0)

        # export (write) fitted parameters to .npz file in tmp directory
        tmp_dir = tempfile.TemporaryDirectory()
        vf.write_npz(tmp_dir.name)

        # create a new vector fitting instance and import (read) those fitted parameters
        vf2 = skrf.vectorFitting.VectorFitting(nw)
        vf2.read_npz(os.path.join(tmp_dir.name, 'coefficients_{}.npz'.format(nw.name)))

        # compare both sets of parameters
        self.assertTrue(np.allclose(vf.poles, vf2.poles))
        self.assertTrue(np.allclose(vf.residues, vf2.residues))
        self.assertTrue(np.allclose(vf.proportional_coeff, vf2.proportional_coeff))
        self.assertTrue(np.allclose(vf.constant_coeff, vf2.constant_coeff))

    #@pytest.mark.skipif(not test_all)
    def test_matplotlib_missing(self):
        vf = skrf.vectorFitting.VectorFitting(skrf.data.ring_slot)
        skrf.vectorFitting.mplt = None
        with self.assertRaises(RuntimeError):
            vf.plot_convergence()

    #@pytest.mark.skipif(not test_all)
    def test_passivity_enforcement(self):
        vf = skrf.VectorFitting(None)

        # non-passive example parameters from Gustavsen's passivity assessment paper:
        vf.poles = np.array([-1, -5 + 6j])
        vf.residues = np.array([[0.3, 4 + 5j], [0.1, 2 + 3j], [0.1, 2 + 3j], [0.4, 3 + 4j]])
        vf.constant_coeff = np.array([0.2, 0.1, 0.1, 0.3])
        vf.proportional_coeff = np.array([0.0, 0.0, 0.0, 0.0])

        # test if model is not passive
        violation_bands = vf.passivity_test(parameter_type='S')
        self.assertTrue(np.allclose(violation_bands, np.array([4.2472, 16.434]) / 2 / np.pi, rtol=1e-3, atol=1e-3))
        self.assertFalse(vf.is_passive())

        # enforce passivity with default settings
        vf.passivity_enforce(parameter_type='S')

        # check if model is now passive
        self.assertTrue(vf.is_passive())

        # verify that perturbed residues are correct
        passive_residues = np.array([[0.11758964+0.j, 2.65059197+3.29414469j],
                                     [-0.06802029+0.j, 0.77242142+1.44226975j],
                                     [-0.06802029+0.j, 0.77242142+1.44226975j],
                                     [0.24516918+0.j, 1.88377719+2.57735204j]])
        self.assertTrue(np.allclose(vf.residues, passive_residues))

    #@pytest.mark.skipif(not test_all)
    def test_passivity_enforcement_hybrid_saraswat(self):
        # Example obtained from the poles and residues in Table 1 of
        # [IEEE Tran. on Adv. Pkg. 27, 57 (2004)]
        poles = np.array([-3.3385, -6.2928, 
                          -1.988 + 1j*1.7803e1,
                          -1.988 - 1j*1.7803e1,
                          -1.6406 + 1j*2.4315e1,
                          -1.6406 - 1j*2.4315e1,
                          -3.6021e-4 + 1j*2.4998e1,
                          -3.6021e-4 - 1j*2.4998e1,
                          -4.0573e-1 + 1j*3.2744e1,
                          -4.0573e-1 - 1j*3.2744e1 ])
        residues = np.array([
            [ 1.2184e-1, -1.2723e-1], 
            [-1.2723e-1, 1.3286e-1 ], 
            [-7.3081e-9, -2.9387e-10], 
            [-2.9387e-10, 8.1838e-10],
            [1.262 + 1j*5.2718e-2, 1.9255e-1 + 1j*3.804e-2], 
            [1.9255e-1 + 1j*3.804e-2, 2.7869e-2 + 1j*1.2787e-2],
            [1.262 - 1j*5.2718e-2, 1.9255e-1 - 1j*3.804e-2], 
            [1.9255e-1 - 1j*3.804e-2, 2.7869e-2 - 1j*1.2787e-2],
            [2.2657e-1 + 1j*9.3384e-1, -1.5918e-1 + 1j*4.1527e-2], 
            [-1.5918e-1 + 1j*4.1527e-2, 2.7689e-2 + 1j*7.5762e-3],
            [2.2657e-1 - 1j*9.3384e-1, -1.5918e-1 - 1j*4.1527e-2], 
            [-1.5918e-1 - 1j*4.1527e-2, 2.7689e-2 - 1j*7.5762e-3],
            [2.8816e-8 + 1j*4.0282e-8, -1.2377e-8 + 1j*5.4798e-9],
            [-1.2377e-8 + 1j*5.4798e-9, 4.547e-10 + 1j*1.3065e-7],
            [2.8816e-8 - 1j*4.0282e-8, -1.2377e-8 - 1j*5.4798e-9],
            [-1.2377e-8 - 1j*5.4798e-9, 4.547e-10 - 1j*1.3065e-7],
            [2.433e-1 + 1j*1.1006e-1, 3.0242e-2 + 1j*1.2548e-2],
            [3.0242e-2 + 1j*1.2548e-2, 3.7552e-3 + 1j*1.4209e-3],
            [2.433e-1 - 1j*1.1006e-1, 3.0242e-2 - 1j*1.2548e-2],
            [3.0242e-2 - 1j*1.2548e-2, 3.7552e-3 - 1j*1.4209e-3]
            ])

        # Constant matrix from the text
        D     = np.array([1.9291e-6, 0., 0., 6.8561e-8])
        freqs = np.linspace(0, 7, 301)
        s     = 1j * 2 * np.pi * freqs

        # Setup the Network and VectorFitting objects
        nw                    = skrf.Network()
        nw.f                  = freqs
        vf                    = skrf.VectorFitting(None)
        vf.poles              = poles
        vf.residues           = residues
        vf.constant_coeff     = D
        vf.proportional_coeff = np.zeros(D.shape)
        vf.network            = nw

        # Compute ABCDE matrices, then S matrix
        A, B, C, D, E = vf._get_ABCDE()
        vf.network.s  = np.array([vf._get_s_from_ABCDE(f, A, B, C, D, E)
                                  for f in freqs])

        # plot the matrix values of the impedance
        Nports = int(np.sqrt(D.size))
        fig, ax = mplt.subplots(Nports, Nports, tight_layout=True)
        for i in range(Nports):
            for j in range(Nports):
                vf.plot_s_db(i, j, freqs=freqs, ax=ax[i][j])
        fig.savefig('zmatrix_mag_saraswat.pdf', format='pdf')

        # test if model is not passive
        violation_bands = vf.passivity_test(parameter_type='Z')

        # enforce passivity with default settings
        vf.passivity_enforce(parameter_type='Z')
        for i in range(Nports):
            for j in range(Nports):
                vf.plot_s_db(i, j, freqs=freqs, ax=ax[i][j])
        fig.savefig('zmatrix_mag_passive_saraswat.pdf', format='pdf')

        # plot the eigenvalues of the Hermitian part of Z to check if there are
        # any passivity violations that went undetected
        fname = 'zH_saraswat.pdf'
        vf.plot_zH_eigenvalues(freqs=freqs, fname=fname)

        # check if model is now passive
        self.assertTrue(vf.is_passive())

    def test_passivity_enforcement_hybrid_mcdermott(self, use_real_imag=False):
        """
        Tests VectorFit and its passivity correction algorithm on impedance
        matrix data generated by HFSS
        """
        # Load the data from file and create a network from the impedance data
        tld = './skrf/data'

        fname1 = f'{tld}/nrm210802_McDermott_Qubit_loss1en5_Qubit1_zmatrix.tab'
        fname2 = f'{tld}/nrm211212_McDermott_loss1en5_Qubit1_real_imag.tab'
        fname = fname2 if use_real_imag else fname1

        # Read the frequencies and impedances in a single shot
        zdata = np.genfromtxt(fname, delimiter='\t', skip_header=2).T
        freqs = zdata[0]
        Nf = freqs.size
        Nports = int(np.sqrt((len(zdata) - 1) // 2))

        print(f'zdata.shape: {zdata.shape}')
        print(f'Nports: {Nports}, Nf: {Nf}')
        print(f'zdata.size: {zdata.size}')
        print(f'freqs: {freqs}')
    
        # Initialize and populate the Z-matrix
        print(f'Reading impedance data from {fname} ...')
        Z = np.zeros([Nf, Nports, Nports], dtype=np.complex128)

        for i in range(Nports):
            for j in range(Nports):
                zidx = 2 * (i * Nports + j) + 1
                # Read real and imaginary data
                if use_real_imag:
                    Zre = zdata[zidx]
                    Zim = zdata[zidx + 1]
                    Z[:, i, j] = Zre + 1j * Zim
                else:
                    # Zmag = 10**(zdata[zidx] / 20.)
                    Zmag = zdata[zidx]
                    Zph  = zdata[zidx + 1]
                    Z[:, i, j] = Zmag * np.exp(1j * Zph * np.pi / 180.)
    
        # Create a network object to pass to VectorFit
        print(f'Constructing network from impedance data ...')
        nw       = skrf.Network.from_z(Z, f=freqs)
        vf       = skrf.vectorFitting.VectorFitting(nw)
        vf.freqs = freqs

        # Check for the data scales on the input
        Zmin = np.min([np.min(np.real(np.linalg.eigvals((Z[k, :, :] 
                            + Z[k, :, :].T.conj()) / 2)))
            for k in range(freqs.size)])
        print(f'Zmin: {Zmin}')

        # # Plot the eigenvalues before vector fit
        # fname = 'zH_data_mcdermott.pdf'
        # print('Plotting eigenvalues of ZH ...')
        # vf.plot_zH_eigenvalues(freqs=freqs, fname=fname)

        print(f'Performing Vector Fit on the network ...')
        # vf.vector_fit(n_poles_real=2, n_poles_cmplx=5, fit_proportional=False,
        #               fit_constant=True)
        vf.vector_fit(n_poles_real=2, n_poles_cmplx=5, fit_proportional=True,
                      fit_constant=True, parameter_type='z')
        # vf.vector_fit(n_poles_real=0, n_poles_cmplx=20, fit_proportional=True,
        #               fit_constant=True, parameter_type='z')

        # Write the VectorFit results to file
        print('Computing ABCDE matrices ...')
        A, B, C, D, E = vf._get_ABCDE()
        fname = 'mcdermott_vf_abcde_211216.hdf5'
        print('Writing ABCDE matrices to file ...')
        skrf.write_to_hdf([freqs, A, B, C, D, E],
                          ['f', 'A', 'B', 'C', 'D', 'E'],
                          fname)
        Zfit = np.array([vf._get_s_from_ABCDE(f, A, B, C, D, E)
                         for f in freqs])
        Zfitmin = np.min([np.min(np.real(np.linalg.eigvals((Zfit[k, :, :] 
                            + Zfit[k, :, :].T.conj()) / 2)))
                            for k in range(freqs.size)])
        print(f'Zfitmin: {Zfitmin}')

        # Check for asymptotic passivity, and refit without negative eigenvalues
        # in the D-matrix
        Deig, UD = np.linalg.eig(D)
        Eeig, UE = np.linalg.eig(E)
        eps = 1e-8
        if np.any(Deig < 0):
            print(f'Deig: {Deig}')
            print(f'Eeig: {Eeig}')
            pidx = np.where(Deig > 0)
            peidx = np.where(Eeig > 0)
            nidx = np.where(Deig < 0)
            neidx = np.where(Eeig < 0)
            Deig_posdef = np.zeros(Deig.size)
            Eeig_posdef = np.zeros(Eeig.size)
            Deig_posdef[pidx] = Deig[pidx]
            Eeig_posdef[peidx] = Eeig[peidx]
            Deig_posdef[nidx]  = eps * np.ones(len(nidx)) 
            Eeig_posdef[neidx] = eps * np.ones(len(neidx))
            D_asym = UD @ np.diag(Deig_posdef) @ np.linalg.inv(UD)
            E_asym = UE @ np.diag(Eeig_posdef) @ np.linalg.inv(UE)
            print(f'Deig_asym: {Deig_posdef}')
            print(f'Eeig_asym: {Eeig_posdef}')

            # Rerun vector fit with a new Z
            Z_asym  = np.array([vf._get_s_from_ABCDE(ff, A, B, C, D_asym,
                                E_asym) for ff in freqs])
            nw_asym = skrf.Network.from_z(Z_asym, f=freqs)
            vf = skrf.vectorFitting.VectorFitting(nw_asym)

            vf.vector_fit(n_poles_real=2, n_poles_cmplx=5,
                    fit_proportional=False,
                      fit_constant=True, parameter_type='z')

        # Check the poles and residues here
        poles_fit    = np.copy(vf.poles)
        residues_fit = np.copy(vf.residues)

        # Plot the eigenvalues after vector fit
        fname = 'zH_vf_mcdermott.pdf'
        print('Plotting eigenvalues of ZH ...')
        vf.plot_zH_eigenvalues(freqs=freqs, fname=fname)
        fname = 'zH_min_vf_mcdermott.pdf'
        vf.plot_zH_eigenvalues(freqs=freqs, fname=fname, plot_min_only=True)

        # plot the real, imaginary components of the impedance
        # real part
        fig, ax = mplt.subplots(Nports, Nports, tight_layout=True)
        for i in range(Nports):
            for j in range(Nports):
                vf.plot_z_db(i, j, freqs=freqs, ax=ax[i, j])
        fig.savefig('zmatrix_mag_db_mcdermott.pdf', format='pdf')
        mplt.close('all')
        fig, ax = mplt.subplots(Nports, Nports, tight_layout=True)
        for i in range(Nports):
            for j in range(Nports):
                vf.plot_z_mag(i, j, freqs=freqs, ax=ax[i, j])
        fig.savefig('zmatrix_mag_mcdermott.pdf', format='pdf')
        mplt.close('all')
        fig, ax = mplt.subplots(Nports, Nports, tight_layout=True)
        for i in range(Nports):
            for j in range(Nports):
                vf.plot_z_re(i, j, freqs=freqs, ax=ax[i, j])
        fig.savefig('zmatrix_re_mcdermott.pdf', format='pdf')
        mplt.close('all')

        # imaginary part
        fig, ax = mplt.subplots(Nports, Nports, tight_layout=True)
        for i in range(Nports):
            for j in range(Nports):
                vf.plot_z_im(i, j, freqs=freqs, ax=ax[i, j])
        fig.savefig('zmatrix_im_mcdermott.pdf', format='pdf')
        mplt.close('all')

        # test if model is not passive
        # vf.vector_fit(n_poles_real=2, n_poles_cmplx=5, fit_proportional=False,
        #               fit_constant=True, parameter_type='z')

        # vf.vector_fit(n_poles_real=0, n_poles_cmplx=20, fit_proportional=False,
        #               fit_constant=True, parameter_type='z')
        print('Computing regions of passivity violations of Z ...')
        violation_bands = vf.passivity_test(parameter_type='Z')

        # enforce passivity with default settings
        print('Applying passivity correction to Z ...')
        # vf.max_iterations = 100
        vf.max_iterations = 20
        # vf.passivity_enforce(parameter_type='Z', n_samples=10*freqs.size) # freqs.size)
        vf.passivity_enforce(parameter_type='Z', n_samples=2*freqs.size,
                            delta_in=1e-2) # freqs.size)

        # write the passive ABCDE matrices to file
        print('Computing passive ABCDE matrices ...')
        A1, B1, C1, D1, E1 = vf._get_ABCDE()
        Deig, UD = np.linalg.eigh(0.5 * (D1 + D1.T.conj()) )
        Eeig, ED = np.linalg.eig(E1)
        print(f'eig (D + D^t)/2: {Deig}')
        print(f'Eeig: {Eeig}')
    
        is_C_changed = np.allclose(C, C1)
        is_R_changed = np.allclose(residues_fit, vf.residues)

        print('Did the passivity update actually happen?')
        print(f'C == C1? {is_C_changed}')
        print(f'R == R1? {is_R_changed}')
        print(f'E1:\n{E1}')

        dstr = '220201'
        fname = f'mcdermott_vf_abcde_{dstr}.hdf5'
        print('Writing ABCDE matrices to file ...')
        skrf.write_to_hdf([freqs, A1, B1, C1, D1, E1],
                          ['f', 'A', 'B', 'C', 'D', 'E'],
                          fname)

        print('Writing poles and residues to file ...')
        dstr = datetime.datetime.today().strftime('%y%m%d')
        fname = f'mcdermott_vf_poles_residues_{dstr}.hdf5'
        skrf.write_to_hdf([freqs, vf.poles, vf.residues, D1], 
                          ['f', 'poles', 'residues', 'D'], fname)

        # get the updated Z matrix and the mininum eigenvalues
        print('Computing the eigenvalues of ZH ...')
        w0 = 8.86688e9
        freqs_log = np.sort(np.concatenate((
            np.logspace(np.log10(np.max([w0,1e-3]))-3,
            np.log10(np.max([w0,1e-3]))+3,1000),
            np.linspace(np .max([w0,1e-3])*.8,
            np.max([w0,1e-3])*1.2,100000))))
        # freqs_log = np.logspace(7, 13, 100001)
        Zfitpass = np.array([vf._get_s_from_ABCDE(f, A1, B1, C1, D1, E1) 
                            for f in freqs_log])
        Zfitpassmin = np.array([np.min(np.real(np.linalg.eigvals((
                Zfitpass[k, :, :] + Zfitpass[k, :, :].T.conj()) / 2)))
                            for k in range(freqs_log.size)])
        # Zfitpassminidx = np.argmin([np.min(np.real(np.linalg.eigvals((
        #         Zfitpass[k, :, :] + Zfitpass[k, :, :].T.conj()) / 2)))
        #                     for k in range(vf.network.f.size)])
        # Zfitpassminfreq = vf.network.f[Zfitpassminidx]
        # print(f'Zfitpassmin(f={Zfitpassminfreq}): {Zfitpassmin}')

        print('Plotting the eigenvalues of ZH ...')
        fig, ax = mplt.subplots(1, 1, tight_layout=True)
        ax.plot(freqs_log, Zfitpassmin)
        ax.plot(freqs_log, np.zeros(freqs_log.size), 'k--')
        ax.set_xscale('log')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel(r'Re $\{\lambda\{Z_H(j\omega)\}\}$')
        ax.set_ylim([-30, 30])
        fig.savefig('zH_min_passive_check.pdf')
        mplt.close('all')

        # plot the eigenvalues of the Hermitian part of Z to check if there are
        # any passivity violations that went undetected
        fname = 'zH_passive_mcdermott.pdf'
        print('Plotting passive eigenvalues of ZH ...')
        vf.plot_zH_eigenvalues(freqs=freqs, fname=fname, ylim=[-0.5, None])
        fname = 'zH_min_passive_mcdermott.pdf'
        vf.plot_zH_eigenvalues(freqs=freqs, fname=fname, ylim=None,
                plot_min_only=True) #[-0.5, 1.3])

        # plot the matrix values of the impedance
        # plot the real, imaginary components of the impedance
        # real part
        fig, ax = mplt.subplots(Nports, Nports, tight_layout=True)
        for i in range(Nports):
            for j in range(Nports):
                vf.plot_z_re(i, j, freqs=freqs, ax=ax[i, j])
        fig.savefig('zmatrix_passive_re_mcdermott.pdf', format='pdf')
        mplt.close('all')
        fig, ax = mplt.subplots(Nports, Nports, tight_layout=True)
        for i in range(Nports):
            for j in range(Nports):
                vf.plot_z_im(i, j, freqs=freqs, ax=ax[i, j])
        fig.savefig('zmatrix_passive_im_mcdermott.pdf', format='pdf')
        mplt.close('all')

        # plot the passivity vs. iteration
        fig, ax = mplt.subplots(1, 1, tight_layout=True)
        vf.plot_passivation(ax=ax)
        fig.savefig('min_eig_vs_iter_mcdermott.pdf', format='pdf')

        # check if model is now passive
        self.assertTrue(vf.is_passive(parameter_type='Z'))

    # #@pytest.mark.skipif(not test_all)
    # def test_vf_pole_count_hybrid_mcdermott(self, use_real_imag=False):
    #     """
    #     Tests VectorFit and its passivity correction algorithm on impedance
    #     matrix data generated by HFSS
    #     """
    #     # Load the data from file and create a network from the impedance data
    #     tld = './skrf/data'
    # 
    #     fname1 = f'{tld}/nrm210802_McDermott_Qubit_loss1en5_Qubit1_zmatrix.tab'
    #     fname2 = f'{tld}/nrm211212_McDermott_loss1en5_Qubit1_real_imag.tab'
    #     fname = fname2 if use_real_imag else fname1
    # 
    #     # Read the frequencies and impedances in a single shot
    #     zdata = np.genfromtxt(fname, delimiter='\t', skip_header=2).T
    #     freqs = zdata[0]
    #     Nf = freqs.size
    #     Nports = int(np.sqrt((len(zdata) - 1) // 2))
    # 
    #     print(f'zdata.shape: {zdata.shape}')
    #     print(f'Nports: {Nports}, Nf: {Nf}')
    #     print(f'zdata.size: {zdata.size}')
    #     print(f'freqs: {freqs}')
    # 
    #     # Initialize and populate the Z-matrix
    #     print(f'Reading impedance data from {fname} ...')
    #     Z = np.zeros([Nf, Nports, Nports], dtype=np.complex128)
    # 
    #     for i in range(Nports):
    #         for j in range(Nports):
    #             zidx = 2 * (i * Nports + j) + 1
    #             # Read real and imaginary data
    #             if use_real_imag:
    #                 Zre = zdata[zidx]
    #                 Zim = zdata[zidx + 1]
    #                 Z[:, i, j] = Zre + 1j * Zim
    #             else:
    #                 Zmag = zdata[zidx]
    #                 Zph = zdata[zidx + 1]
    #                 Z[:, i, j] = Zmag * np.exp(1j * Zph)
    # 
    #     # Create a network object to pass to VectorFit
    #     print(f'Constructing network from impedance data ...')
    #     nw       = skrf.Network.from_z(Z, f=freqs)
    #     vf       = skrf.vectorFitting.VectorFitting(nw)
    #     vf.freqs = freqs
    # 
    #     # Set the number of poles on each iteration
    #     Mp = np.linspace(0, 50, 51)
    #     rms_errs = np.zeros(len(Mp))
    # 
    #     # Iterate over increasing number of complex pole pairs
    #     for i in range(len(Mp)):
    #         print(f'Number of poles: {i+3} ...')
    #         vf.vector_fit(n_poles_real=0, n_poles_cmplx=3+i,
    #                       fit_proportional=True, fit_constant=True,
    #                       parameter_type='z')
    #         rms_errs[i] = np.max(vf.get_rms_error(i=list(range(Nports)), 
    #                             j=list(range(Nports)), parameter_type='z'))
    # 
    #     # Plot the results of the errors
    #     fig, ax = mplt.subplots(1, 1, tight_layout=True)
    #     ax.plot(Mp+3, rms_errs, ls='-', marker='o')
    #     fsize = 20
    #     ax.set_yscale('log')
    #     ax.set_xlabel('No. of complex pole pairs', fontsize=fsize)
    #     ax.set_ylabel('Max Element-wise RMS Error', fontsize=fsize)
    #     fig.savefig('zmatrix_rms_errs_mcdermott.pdf', format='pdf')
    #     mplt.close('all')


suite = unittest.TestLoader().loadTestsFromTestCase(VectorFittingTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)
