import unittest
import skrf
import numpy as np
import tempfile
import os
import warnings
import matplotlib.pyplot as mplt


class VectorFittingTestCase(unittest.TestCase):

    def test_ringslot_with_proportional(self):
        # perform the fit
        nw = skrf.data.ring_slot
        vf = skrf.vectorFitting.VectorFitting(nw)
        vf.vector_fit(n_poles_real=2, n_poles_cmplx=0, fit_proportional=True, fit_constant=True)
        self.assertLess(vf.get_rms_error(), 0.02)

    def test_ringslot_default_log(self):
        # perform the fit without proportional term
        nw = skrf.data.ring_slot
        vf = skrf.vectorFitting.VectorFitting(nw)
        vf.vector_fit(n_poles_real=4, n_poles_cmplx=0, init_pole_spacing='log')
        self.assertLess(vf.get_rms_error(), 0.01)

    def test_ringslot_without_prop_const(self):
        # perform the fit without proportional term
        nw = skrf.data.ring_slot
        vf = skrf.vectorFitting.VectorFitting(nw)
        vf.vector_fit(n_poles_real=4, n_poles_cmplx=0, fit_proportional=False, fit_constant=False)
        self.assertLess(vf.get_rms_error(), 0.01)

    def test_190ghz_measured(self):
        # perform the fit without proportional term
        nw = skrf.network.Network('./doc/source/examples/vectorfitting/190ghz_tx_measured.S2P')
        vf = skrf.vectorFitting.VectorFitting(nw)
        vf.vector_fit(n_poles_real=4, n_poles_cmplx=4, fit_proportional=False, fit_constant=True)
        self.assertLess(vf.get_rms_error(), 0.02)

    def test_no_convergence(self):
        # perform a bad fit that does not converge and check if a RuntimeWarning is given
        with warnings.catch_warnings(record=True) as warning:
            nw = skrf.network.Network('./doc/source/examples/vectorfitting/190ghz_tx_measured.S2P')
            vf = skrf.vectorFitting.VectorFitting(nw)
            vf.vector_fit(n_poles_real=0, n_poles_cmplx=5, fit_proportional=False, fit_constant=True)
            self.assertEqual(warning[-1].category, RuntimeWarning)

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

    def test_matplotlib_missing(self):
        vf = skrf.vectorFitting.VectorFitting(skrf.data.ring_slot)
        skrf.vectorFitting.mplt = None
        with self.assertRaises(RuntimeError):
            vf.plot_convergence()

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
                    Zmag = zdata[zidx]
                    Zph = zdata[zidx + 1]
                    Z[:, i, j] = Zmag * np.exp(1j * Zph)
    
        # Create a network object to pass to VectorFit
        print(f'Constructing network from impedance data ...')
        nw       = skrf.Network.from_z(Z, f=freqs)
        vf       = skrf.vectorFitting.VectorFitting(nw)
        vf.freqs = freqs

        # # Plot the eigenvalues before vector fit
        # fname = 'zH_data_mcdermott.pdf'
        # print('Plotting eigenvalues of ZH ...')
        # vf.plot_zH_eigenvalues(freqs=freqs, fname=fname)

        print(f'Performing Vector Fit on the network ...')
        # vf.vector_fit(n_poles_real=2, n_poles_cmplx=5, fit_proportional=False,
        #               fit_constant=True)
        vf.vector_fit(n_poles_real=2, n_poles_cmplx=5, fit_proportional=True,
                      fit_constant=True, parameter_type='z')

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

        # Plot the eigenvalues after vector fit
        fname = 'zH_vf_mcdermott.pdf'
        print('Plotting eigenvalues of ZH ...')
        vf.plot_zH_eigenvalues(freqs=freqs, fname=fname)

        # plot the real, imaginary components of the impedance
        # real part
        fig, ax = mplt.subplots(Nports, Nports, tight_layout=True)
        for i in range(Nports):
            for j in range(Nports):
                vf.plot_z_db(i, j, freqs=freqs, ax=ax[i, j])
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
        vf.vector_fit(n_poles_real=2, n_poles_cmplx=5, fit_proportional=False,
                      fit_constant=True, parameter_type='z')
        print('Computing regions of passivity violations of Z ...')
        violation_bands = vf.passivity_test(parameter_type='Z')

        # enforce passivity with default settings
        print('Applying passivity correction to Z ...')
        vf.passivity_enforce(parameter_type='Z')

        # plot the eigenvalues of the Hermitian part of Z to check if there are
        # any passivity violations that went undetected
        fname = 'zH_passive_mcdermott.pdf'
        print('Plotting passive eigenvalues of ZH ...')
        vf.plot_zH_eigenvalues(freqs=freqs, fname=fname)

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

        # check if model is now passive
        self.assertTrue(vf.is_passive())

    def test_vf_pole_count_hybrid_mcdermott(self, use_real_imag=False):
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
                    Zmag = zdata[zidx]
                    Zph = zdata[zidx + 1]
                    Z[:, i, j] = Zmag * np.exp(1j * Zph)
    
        # Create a network object to pass to VectorFit
        print(f'Constructing network from impedance data ...')
        nw       = skrf.Network.from_z(Z, f=freqs)
        vf       = skrf.vectorFitting.VectorFitting(nw)
        vf.freqs = freqs

        # Set the number of poles on each iteration
        Mp = np.linspace(0, 50, 51)
        rms_errs = np.zeros(len(Mp))

        # Iterate over increasing number of complex pole pairs
        for i in range(len(Mp)):
            print(f'Number of poles: {i+3} ...')
            vf.vector_fit(n_poles_real=0, n_poles_cmplx=3+i,
                          fit_proportional=True, fit_constant=True,
                          parameter_type='z')
            rms_errs[i] = np.max(vf.get_rms_error(i=list(range(Nports)), 
                                j=list(range(Nports)), parameter_type='z'))

        # Plot the results of the errors
        fig, ax = mplt.subplots(1, 1, tight_layout=True)
        ax.plot(Mp+3, rms_errs, ls='-', marker='o')
        fsize = 20
        ax.set_yscale('log')
        ax.set_xlabel('No. of complex pole pairs', fontsize=fsize)
        ax.set_ylabel('Max Element-wise RMS Error', fontsize=fsize)
        fig.savefig('zmatrix_rms_errs_mcdermott.pdf', format='pdf')
        mplt.close('all')


suite = unittest.TestLoader().loadTestsFromTestCase(VectorFittingTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)
